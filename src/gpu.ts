export type GpuMesh = {
  positions: Float32Array;
  indices: Uint32Array;
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
  private computeGroup?: GPUBindGroup;
  private renderGroup?: GPUBindGroup;
  private depth?: GPUTexture;
  private cells = 0;
  private center = new Float32Array(4);
  private radius = 1;
  private readback: GPUBuffer;
  private reading = false;
  private destroyed = false;
  private generation = 0;
  private params: GPUBuffer;
  private camera: GPUBuffer;
  private counters: GPUBuffer;
  private observer: ResizeObserver;
  private scene?: Scene;
  private computeDirty = false;
  private needStats = false;
  onStats?: (stats: Stats) => void;
  onError?: (error: string) => void;
  private constructor(
    private canvas: HTMLCanvasElement,
    private device: GPUDevice,
    private context: GPUCanvasContext,
    private compute: GPUComputePipeline,
    private render: GPURenderPipeline,
  ) {
    this.params = device.createBuffer({
      size: 112,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.camera = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.counters = device.createBuffer({
      size: 20,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
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
    const count = mesh.indices.length / 8;
    const outputSize = Math.max(32, count * 6 * 32);
    if (
      outputSize > this.device.limits.maxStorageBufferBindingSize ||
      outputSize > this.device.limits.maxBufferSize
    )
      throw new Error(
        `GPU メモリ上限を超えています。この環境では最大 ${Math.floor(Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / 192).toLocaleString()} 胞まで読み込めます。`,
      );
    if (
      Math.ceil(count / 64) >
      this.device.limits.maxComputeWorkgroupsPerDimension
    )
      throw new Error("GPU dispatch limit exceeded");
    for (const data of [mesh.positions, mesh.indices])
      if (data.byteLength > this.device.limits.maxStorageBufferBindingSize)
        throw new Error("Input exceeds GPU storage buffer limit");
    const upload = (data: Float32Array | Uint32Array) => {
      const buffer = this.device.createBuffer({
        size: Math.max(16, data.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        buffer,
        0,
        data as GPUAllowSharedBufferSource,
      );
      return buffer;
    };
    const positions = upload(mesh.positions),
      indices = upload(mesh.indices);
    const output = this.device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.STORAGE,
    });
    const computeGroup = this.device.createBindGroup({
      layout: this.compute.getBindGroupLayout(0),
      entries: [positions, indices, output, this.counters, this.params].map(
        (buffer, binding) => ({ binding, resource: { buffer } }),
      ),
    });
    const renderGroup = this.device.createBindGroup({
      layout: this.render.getBindGroupLayout(0),
      entries: [output, this.camera].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    this.meshBuffers.forEach((b) => b.destroy());
    this.meshBuffers = [positions, indices, output];
    this.computeGroup = computeGroup;
    this.renderGroup = renderGroup;
    this.cells = count;
    this.center = new Float32Array(mesh.center);
    this.radius = Math.max(mesh.radius, 1e-30);
    this.generation++;
  }
  draw(scene: Scene, recompute = true): void {
    if (this.destroyed || !this.computeGroup || !this.renderGroup) return;
    this.scene = scene;
    this.computeDirty ||= recompute;
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
    const computeNow = this.computeDirty;
    this.computeDirty = false;
    if (computeNow) {
      this.needStats = true;
      const bytes = new ArrayBuffer(112);
      const floats = new Float32Array(bytes);
      floats.set(rotation4(scene.angles));
      floats.set(this.center, 16);
      floats.set(
        [scene.offset, scene.axis, this.radius * 1e-6, this.radius],
        20,
      );
      new Uint32Array(bytes).set([this.cells, 0, 0, 0], 24);
      this.device.queue.writeBuffer(this.params, 0, bytes);
      this.device.queue.writeBuffer(
        this.counters,
        0,
        new Uint32Array([0, 1, 0, 0, 0]),
      );
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.compute);
      pass.setBindGroup(0, this.computeGroup);
      pass.dispatchWorkgroups(Math.ceil(this.cells / 64));
      pass.end();
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.025, g: 0.036, b: 0.055, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depth!.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.render);
    pass.setBindGroup(0, this.renderGroup);
    pass.drawIndirect(this.counters, 0);
    pass.end();
    const read = this.needStats && !this.reading;
    const generation = this.generation;
    if (read) {
      this.needStats = false;
      this.reading = true;
      encoder.copyBufferToBuffer(this.counters, 0, this.readback, 0, 20);
    }
    this.device.queue.submit([encoder.finish()]);
    if (read) {
      void this.readback
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          if (this.destroyed) return;
          const counts = new Uint32Array(this.readback.getMappedRange());
          const stats = {
            triangles: counts[0] / 3,
            coplanar: counts[4],
            milliseconds: performance.now() - started,
          };
          this.readback.unmap();
          if (generation === this.generation) this.onStats?.(stats);
        })
        .catch((error) => {
          if (!this.destroyed) this.onError?.(String(error));
        })
        .finally(() => {
          this.reading = false;
          if (this.needStats && this.scene && !this.destroyed)
            this.draw(this.scene, false);
        });
    }
  }
  destroy(): void {
    this.destroyed = true;
    this.observer.disconnect();
    this.meshBuffers.forEach((b) => b.destroy());
    this.depth?.destroy();
    this.params.destroy();
    this.camera.destroy();
    this.counters.destroy();
    this.readback.destroy();
    this.context.unconfigure();
    this.device.destroy();
  }
}
