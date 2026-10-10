export type GpuMesh = {
  positions: Float32Array;
  indices: Uint32Array;
  vertexNormals?: Float32Array;
  cellNormals?: Float32Array;
  center: Float64Array;
  radius: number;
};
export type Scene = {
  axis: number;
  offset: number;
  angles: number[];
  yaw: number;
  pitch: number;
  distance: number;
  style: number;
  smooth: boolean;
};
export type Stats = {
  triangles: number;
  coplanar: number;
  milliseconds: number;
};

export function rotation4(angles: number[]): Float32Array {
  const m = new Float32Array(16);
  for (let i = 0; i < 4; i++) m[i * 4 + i] = 1;
  const planes = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
  ];
  planes.forEach(([a, b], k) => {
    const c = Math.cos(angles[k]),
      s = Math.sin(angles[k]);
    for (let j = 0; j < 4; j++) {
      const x = m[j * 4 + a],
        y = m[j * 4 + b];
      m[j * 4 + a] = c * x - s * y;
      m[j * 4 + b] = s * x + c * y;
    }
  });
  return m;
}
export class Renderer {
  private meshBuffers: GPUBuffer[] = [];
  private batches: { flags: number; count: number; params: GPUBuffer; counters: GPUBuffer; computeGroup: GPUBindGroup }[] = [];
  private renderGroup?: GPUBindGroup;
  private depth?: GPUTexture;
  private center = new Float32Array(4);
  private radius = 1;
  private readback: GPUBuffer;
  private reading = false;
  private destroyed = false;
  private generation = 0;
  private camera: GPUBuffer;
  private observer: ResizeObserver;
  private scene?: Scene;
  private drawing = false;
  private pendingDraw = false;
  onStats?: (stats: Stats) => void;
  onError?: (error: string) => void;
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private compute: GPUComputePipeline;
  private render: GPURenderPipeline;
  private constructor(
    canvas: HTMLCanvasElement, device: GPUDevice, context: GPUCanvasContext,
    compute: GPUComputePipeline, render: GPURenderPipeline,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.compute = compute;
    this.render = render;
    this.camera = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.observer = new ResizeObserver(() => {
      if (this.scene) this.draw(this.scene, false);
    });
    this.observer.observe(canvas);
    device.lost.then((info) => {
      if (!this.destroyed)
        this.onError?.(`GPU device lost: ${info.message}. Reload the page.`);
    });
    device.addEventListener("uncapturederror", (event) =>
      this.onError?.(event.error.message),
    );
  }
  static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu)
      throw new Error(
        "WebGPU が利用できません。最新版の Chrome / Edge と、ハードウェアアクセラレーションを確認してください。",
      );
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("WebGPU adapter が見つかりません。");
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("WebGPU canvas の初期化に失敗しました。");
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    const sources = await Promise.all(
      ["section", "render"].map(async (name) => {
        const response = await fetch(`${base}/shaders/${name}.wgsl`);
        if (!response.ok) throw new Error(`Shader fetch failed: ${name}`);
        return response.text();
      }),
    );
    const modules = sources.map((code) => device.createShaderModule({ code }));
    for (const module of modules) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((x) => x.type === "error");
      if (errors.length)
        throw new Error(errors.map((x) => x.message).join("\n"));
    }
    const compute = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: modules[0], entryPoint: "main" },
    });
    const render = await device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module: modules[1], entryPoint: "vs" },
      fragment: {
        module: modules[1],
        entryPoint: "fs",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
    context.configure({
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: "opaque",
    });
    return new Renderer(canvas, device, context, compute, render);
  }
  load(mesh: GpuMesh): void {
    // Each batch remaps its vertices, so neither input nor output depends on
    // the total model size. Only one worst-case section buffer is allocated.
    const limit = Math.min(this.device.limits.maxStorageBufferBindingSize,
      this.device.limits.maxBufferSize, 32 * 1024 * 1024);
    const capacity = Math.floor(Math.min(limit / 192,
      this.device.limits.maxComputeWorkgroupsPerDimension * 64));
    if (capacity < 1) throw new Error("GPU buffer capacity is insufficient");
    const count = mesh.indices.length / 8;
    const hasVertexNormals = !!mesh.vertexNormals?.length;
    const hasCellNormals = !!mesh.cellNormals?.length;
    if (hasVertexNormals && mesh.vertexNormals!.length !== mesh.positions.length)
      throw new Error("Vertex normal count mismatch");
    if (hasCellNormals && mesh.cellNormals!.length !== count * 4)
      throw new Error("Cell normal count mismatch");
    const flags = Number(hasVertexNormals) | (Number(hasCellNormals) << 1);
    const buffers: GPUBuffer[] = [];
    const upload = (data: Float32Array | Uint32Array) => {
      const buffer = this.device.createBuffer({ size: Math.max(16, data.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      buffers.push(buffer);
      this.device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
      return buffer;
    };
    const output = this.device.createBuffer({
      size: Math.max(32, Math.min(count, capacity) * 192),
      usage: GPUBufferUsage.STORAGE });
    buffers.push(output);
    const batches: typeof this.batches = [];
    try {
      for (let start = 0; start < count; start += capacity) {
        const size = Math.min(capacity, count - start);
        const ids = new Map<number, number>();
        const packed = mesh.indices.slice(start * 8, (start + size) * 8);
        const points: number[] = [];
        const normals: number[] = [];
        for (let cell = 0; cell < size; cell++) {
          for (let corner = 0; corner < 4; corner++) {
            const at = cell * 8 + corner;
            const original = packed[at];
            let local = ids.get(original);
            if (local === undefined) {
              local = ids.size;
              ids.set(original, local);
              for (let axis = 0; axis < 4; axis++) {
                points.push(mesh.positions[original * 4 + axis]);
                if (hasVertexNormals) normals.push(mesh.vertexNormals![original * 4 + axis]);
              }
            }
            packed[at] = local;
          }
        }
        const positions = upload(new Float32Array(points));
        const indices = upload(packed);
        const originalIds = upload(new Uint32Array(ids.keys()));
        const vertexNormals = upload(hasVertexNormals ? new Float32Array(normals) : new Float32Array(4));
        const cellNormals = upload(hasCellNormals
          ? mesh.cellNormals!.subarray(start * 4, (start + size) * 4) : new Float32Array(4));
        const params = this.device.createBuffer({ size: 112,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const counters = this.device.createBuffer({ size: 20,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT |
            GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        buffers.push(params, counters);
        const computeGroup = this.device.createBindGroup({
          layout: this.compute.getBindGroupLayout(0),
          entries: [positions, indices, output, counters, params, vertexNormals, cellNormals, originalIds].map(
            (buffer, binding) => ({ binding, resource: { buffer } })),
        });
        batches.push({ flags, count: size, params, counters, computeGroup });
      }
      const renderGroup = this.device.createBindGroup({
        layout: this.render.getBindGroupLayout(0),
        entries: [output, this.camera].map((buffer, binding) => ({
          binding, resource: { buffer } })),
      });
      const readback = this.device.createBuffer({ size: Math.max(20, batches.length * 20),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.meshBuffers.forEach((buffer) => buffer.destroy());
      this.readback.destroy();
      this.readback = readback;
      this.meshBuffers = buffers;
      this.batches = batches;
      this.renderGroup = renderGroup;
      this.center = new Float32Array(mesh.center);
      this.radius = Math.max(mesh.radius, 1e-30);
      this.generation++;
    } catch (error) {
      buffers.forEach((buffer) => buffer.destroy());
      throw error;
    }
  }
  draw(scene: Scene, _recompute = true): void {
    if (this.destroyed || !this.batches.length || !this.renderGroup) return;
    this.scene = scene;
    if (this.drawing) { this.pendingDraw = true; return; }
    this.drawing = true;
    this.pendingDraw = false;
    try { this.submit(scene); }
    catch (error) {
      this.drawing = false;
      this.onError?.(String(error));
    }
  }
  private submit(scene: Scene): void {
    if (this.destroyed || !this.batches.length || !this.renderGroup) return;
    this.scene = scene;
    const width = Math.max(
      1,
      Math.min(
        this.device.limits.maxTextureDimension2D,
        Math.round(this.canvas.clientWidth * Math.min(devicePixelRatio, 2)),
      ),
    );
    const height = Math.max(
      1,
      Math.min(
        this.device.limits.maxTextureDimension2D,
        Math.round(this.canvas.clientHeight * Math.min(devicePixelRatio, 2)),
      ),
    );
    if (
      this.canvas.width !== width ||
      this.canvas.height !== height ||
      !this.depth
    ) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        size: [width, height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    this.device.queue.writeBuffer(
      this.camera,
      0,
      new Float32Array([
        width / height,
        scene.yaw,
        scene.pitch,
        scene.distance,
        scene.style,
        scene.axis / 3,
        0,
        0,
      ]),
    );
    const encoder = this.device.createCommandEncoder();
    const started = performance.now();
    // Output storage is reused between batches. Compute and draw remain on GPU;
    // camera-only changes recompute as well because earlier batches are not cached.
    const read = !this.reading;
    const readback = this.readback;
    const generation = this.generation;
    if (read) this.reading = true;
    const view = this.context.getCurrentTexture().createView();
    this.batches.forEach((batch, index) => {
      const bytes = new ArrayBuffer(112);
      const floats = new Float32Array(bytes);
      floats.set(rotation4(scene.angles));
      floats.set(this.center, 16);
      floats.set([scene.offset, scene.axis, this.radius * 1e-6, this.radius], 20);
      new Uint32Array(bytes).set([batch.count, scene.smooth ? batch.flags : batch.flags & 2, 0, 0], 24);
      this.device.queue.writeBuffer(batch.params, 0, bytes);
      this.device.queue.writeBuffer(batch.counters, 0, new Uint32Array([0, 1, 0, 0, 0]));
      const compute = encoder.beginComputePass();
      compute.setPipeline(this.compute);
      compute.setBindGroup(0, batch.computeGroup);
      compute.dispatchWorkgroups(Math.ceil(batch.count / 64));
      compute.end();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view,
          clearValue: { r: 0.025, g: 0.036, b: 0.055, a: 1 },
          loadOp: index === 0 ? "clear" : "load", storeOp: "store" }],
        depthStencilAttachment: { view: this.depth!.createView(),
          depthClearValue: 1, depthLoadOp: index === 0 ? "clear" : "load",
          depthStoreOp: "store" },
      });
      pass.setPipeline(this.render);
      pass.setBindGroup(0, this.renderGroup!);
      pass.drawIndirect(batch.counters, 0);
      pass.end();
      if (read) encoder.copyBufferToBuffer(batch.counters, 0, readback, index * 20, 20);
    });
    this.device.queue.submit([encoder.finish()]);
    void this.device.queue.onSubmittedWorkDone()
      .catch((error) => { if (!this.destroyed) this.onError?.(String(error)); })
      .finally(() => {
        this.drawing = false;
        if (this.pendingDraw && this.scene && !this.destroyed)
          this.draw(this.scene);
      });
    if (read) {
      void readback
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          if (this.destroyed) return;
          const counts = new Uint32Array(readback.getMappedRange());
          let vertices = 0, coplanar = 0;
          for (let at = 0; at < counts.length; at += 5) {
            vertices += counts[at]; coplanar += counts[at + 4];
          }
          const stats = {
            triangles: vertices / 3,
            coplanar,
            milliseconds: performance.now() - started,
          };
          readback.unmap();
          if (generation === this.generation) this.onStats?.(stats);
        })
        .catch((error) => {
          if (!this.destroyed && generation === this.generation) this.onError?.(String(error));
        })
        .finally(() => {
          this.reading = false;
        });
    }
  }
  destroy(): void {
    this.destroyed = true;
    this.observer.disconnect();
    this.meshBuffers.forEach((b) => b.destroy());
    this.depth?.destroy();
    this.camera.destroy();
    this.readback.destroy();
    this.context.unconfigure();
    this.device.destroy();
  }
}
