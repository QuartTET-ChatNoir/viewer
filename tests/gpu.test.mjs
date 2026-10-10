import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { create, globals } from "webgpu";
import init, { ViewerModel } from "../public/wasm/quarttet_core.js";
Object.assign(globalThis, globals);
import { buildBvh } from "../src/bvh.ts";
let gpu, device, pipeline, cullPipeline, preparePipeline;
before(async () => {
  await init({
    module_or_path: await readFile(
      new URL("../public/wasm/quarttet_core_bg.wasm", import.meta.url),
    ),
  });
  gpu = create([
    `backend=${process.env.WEBGPU_BACKEND ?? "vulkan"}`,
    "enable-dawn-features=allow_unsafe_apis",
  ]);
  const adapter = await gpu.requestAdapter();
  assert.ok(
    adapter,
    "GPU adapter required (Linux CI: install mesa-vulkan-drivers)",
  );
  device = await adapter.requestDevice();
  const module = device.createShaderModule({
    code: await readFile(
      new URL("../public/shaders/section.wgsl", import.meta.url),
      "utf8",
    ),
  });
  const info = await module.getCompilationInfo();
  assert.deepEqual(
    info.messages.filter((x) => x.type === "error"),
    [],
  );
  pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

 const cullModule=device.createShaderModule({code:await readFile(new URL("../public/shaders/cull.wgsl",import.meta.url),"utf8")});
 assert.deepEqual((await cullModule.getCompilationInfo()).messages.filter(x=>x.type==="error"),[]);
 cullPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:cullModule,entryPoint:"main"}});
 preparePipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:cullModule,entryPoint:"prepare"}});
});
after(() => {
  device?.destroy();
  gpu = undefined;
});
const planes = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
];
function rotation(angles) {
  const m = new Float32Array(16);
  for (let i = 0; i < 4; i++) m[i * 4 + i] = 1;
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
async function run(model, axis, offset, angles, options={}) {
  const resources = [];
  const buffer = (size, usage) => {
    const b = device.createBuffer({ size, usage });
    resources.push(b);
    return b;
  };
  try {
    const positions = model.positions(),
      original = model.gpu_cells(),
      count = model.cell_count();
    const bvh=buildBvh(positions,original);
    const indices=new Uint32Array(original.length);
    bvh.order.forEach((id,i)=>indices.set(original.subarray(id*8,id*8+8),i*8));
    const input = (data) => {
      const b = buffer(
        data.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      device.queue.writeBuffer(b, 0, data);
      return b;
    };
    const output = buffer(
      count * 384,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const counters = buffer(
      36,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.INDIRECT,
    );
    const uniform = buffer(
      112,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const raw = new ArrayBuffer(112);
    const f = new Float32Array(raw);
    f.set(rotation(angles));
    f.set(model.center(), 16);
    f.set([options.mode ? (options.distance??3)*model.radius() : offset, axis, model.radius() * 1e-6, model.radius()], 20);
    const vertexNormals = model.vertex_normals();
    const sourceNormals = model.cell_normals();
    const cellNormals=new Float32Array(sourceNormals.length);
    bvh.order.forEach((id,i)=>cellNormals.set(sourceNormals.subarray(id*4,id*4+4),i*4));
    const flags = Number(vertexNormals.length > 0) | (Number(cellNormals.length > 0) << 1);
    new Uint32Array(raw).set([count, flags | (options.bvh?4:0), options.mode??0, bvh.roots], 24);
    device.queue.writeBuffer(uniform, 0, raw);
    device.queue.writeBuffer(counters, 0, new Uint32Array([0, 1, 0, 0, 0, 0, 0, 1, 1]));
    const dispatchArgs=buffer(12,GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT);
    const candidates=input(new Uint32Array(count));
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        input(positions),
        input(indices),
        output,
        counters,
        uniform,
        input(vertexNormals.length ? vertexNormals : new Float32Array(4)),
        input(cellNormals.length ? cellNormals : new Float32Array(4)),
        input(Uint32Array.from({length: positions.length / 4}, (_,i) => i)),
        candidates,
      ].map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const read = buffer(
      count * 384 + 64,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const started=performance.now();
    const encoder = device.createCommandEncoder();
    if(options.bvh){
      const cg=device.createBindGroup({layout:cullPipeline.getBindGroupLayout(0),entries:[input(bvh.nodes),candidates,counters,uniform].map((b,binding)=>({binding,resource:{buffer:b}}))});
      const pg=device.createBindGroup({layout:preparePipeline.getBindGroupLayout(0),entries:[{binding:2,resource:{buffer:counters}},{binding:4,resource:{buffer:dispatchArgs}}]});
      const cp=encoder.beginComputePass();cp.setPipeline(cullPipeline);cp.setBindGroup(0,cg);cp.dispatchWorkgroups(Math.ceil(bvh.roots/64));cp.end();
      const pp=encoder.beginComputePass();pp.setPipeline(preparePipeline);pp.setBindGroup(0,pg);pp.dispatchWorkgroups(1);pp.end();
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    if(options.bvh)pass.dispatchWorkgroupsIndirect(dispatchArgs,0);
    else pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(counters, 0, read, 0, 36);
    encoder.copyBufferToBuffer(output, 0, read, 64, count * 384);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const mapped = read.getMappedRange();
    const values = new Uint32Array(mapped, 0, 9).slice();
    const vertices = new Float32Array(mapped, 64, values[0] * 8).slice();
    read.unmap();
    return { values, vertices, elapsed:performance.now()-started };
  } finally {
    resources.forEach((x) => x.destroy());
  }
}
function area(data, stride) {
  let sum = 0;
  for (let i = 0; i < data.length; i += stride * 3) {
    const a = [0, 1, 2].map((j) => data[i + stride + j] - data[i + j]);
    const b = [0, 1, 2].map((j) => data[i + stride * 2 + j] - data[i + j]);
    sum +=
      Math.hypot(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ) / 2;
  }
  return sum;
}
for (let kind = 0; kind < 3; kind++)
  for (let axis = 0; axis < 4; axis++)
    test(`GPU/CPU geometry sample=${kind} axis=${axis}`, async () => {
      const model = ViewerModel.sample(kind);
      try {
        for (const angles of [
          [0, 0, 0, 0, 0, 0],
          [0.11, -0.32, 0.27, 0.19, 0.41, -0.13],
        ])
          for (const offset of [0, 0.173, 2.1]) {
            const cpu = model.slice(axis, offset, new Float64Array(angles));
            const gpu = await run(model, axis, offset, angles);
            assert.equal(
              gpu.values[0],
              cpu.length / 6,
              "GPU vertex count agrees with CPU",
            );
            assert.ok(
              Math.abs(area(gpu.vertices, 8) - area(cpu, 6)) < 2e-5,
              "GPU section surface area agrees with CPU",
            );
            assert.ok(
              gpu.vertices.every(Number.isFinite),
              "finite coordinates and normals",
            );
            const points = (data, stride) => {
              const result = [];
              for (let i = 0; i < data.length; i += stride)
                result.push([data[i], data[i + 1], data[i + 2]]);
              return result;
            };
            const cpuPoints = points(cpu, 6),
              gpuPoints = points(gpu.vertices, 8);
            const contained = (a, b) =>
              a.every((p) =>
                b.some((q) => Math.hypot(...p.map((v, j) => v - q[j])) < 1e-5),
              );
            assert.ok(
              contained(cpuPoints, gpuPoints) &&
                contained(gpuPoints, cpuPoints),
              "GPU/CPU intersection points agree",
            );
            // Compare the actual triangles, independently of workgroup output ordering/winding.
            const canonical = (data, stride) => {
              const list = [];
              for (let i = 0; i < data.length; i += stride * 3) {
                const tri = [];
                for (let k = 0; k < 3; k++)
                  tri.push(
                    [0, 1, 2]
                      .map((j) => Math.round(data[i + k * stride + j] * 1e4))
                      .join(","),
                  );
                list.push(tri.sort().join("|"));
              }
              return list.sort();
            };
            // A quad may use the opposite diagonal on CPU/GPU due to floating-point ordering;
            // both triangulations must still reproduce the same surface area.
            if (cpu.length === 18)
              assert.deepEqual(canonical(gpu.vertices, 8), canonical(cpu, 6));
          }
      } finally {
        model.free();
      }
    });
test("coplanar tetrahedra are reported", async () => {
  const m = ViewerModel.sample(0);
  try {
    const r = await run(m, 3, 1, [0, 0, 0, 0, 0, 0]);
    assert.equal(r.values[4], 6);
  } finally {
    m.free();
  }
});
test("render shader and indirect draw are valid", async () => {
  const module = device.createShaderModule({
    code: await readFile(
      new URL("../public/shaders/render.wgsl", import.meta.url),
      "utf8",
    ),
  });
  const info = await module.getCompilationInfo();
  assert.deepEqual(
    info.messages.filter((x) => x.type === "error"),
    [],
  );
  const pipeline = await device.createRenderPipelineAsync({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });
  const geometry = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    geometry,
    0,
    new Float32Array([
      -0.5, -0.5, 0, 1, 0, 0, 1, 0, 0.5, -0.5, 0, 1, 0, 0, 1, 0, 0, 0.5, 0, 1,
      0, 0, 1, 0,
    ]),
  );
  const camera = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    camera,
    0,
    new Float32Array([1, 0, 0, 2.8, 0, 0, 0, 0]),
  );
  const indirect = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indirect, 0, new Uint32Array([3, 1, 0, 0]));
  const color = device.createTexture({
    size: [64, 64],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    size: [64, 64],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const read = device.createBuffer({
    size: 64 * 256,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [geometry, camera].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: color.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.drawIndirect(indirect, 0);
    pass.end();
    encoder.copyTextureToBuffer(
      { texture: color },
      { buffer: read, bytesPerRow: 256 },
      [64, 64],
    );
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(read.getMappedRange());
    assert.ok(pixels[(32 * 64 + 32) * 4 + 1] > 0, "triangle is rasterized");
    read.unmap();
  } finally {
    [geometry, camera, indirect, read].forEach((x) => x.destroy());
    color.destroy();
    depth.destroy();
  }
});

test("batched renderer preserves the section, depth and statistics", async () => {
  const { Renderer } = await import("../src/gpu.ts");
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  globalThis.devicePixelRatio = 1;
  const renderModule = device.createShaderModule({ code: await readFile(
    new URL("../public/shaders/render.wgsl", import.meta.url), "utf8") });
  const renderPipeline = await device.createRenderPipelineAsync({ layout: "auto",
    vertex: { module: renderModule, entryPoint: "vs" },
    fragment: { module: renderModule, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" } });
  const model = ViewerModel.sample(0);
  const mesh = { positions: model.positions(), indices: model.gpu_cells(),
    vertexNormals: model.vertex_normals(), cellNormals: model.cell_normals(),
    center: model.center(), radius: model.radius() };
  async function draw(limit, offset, smooth, mode=0, acceleration=true) {
    const color = device.createTexture({ size: [64, 64], format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const proxy = new Proxy(device, { get(target, key) {
      if (key === "limits") return { maxStorageBufferBindingSize: limit,
        maxBufferSize: limit, maxComputeWorkgroupsPerDimension: 65535,
        maxTextureDimension2D: 8192 };
      if (key === "destroy") return () => {};
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    }});
    const renderer = new Renderer({ clientWidth: 64, clientHeight: 64, width: 0, height: 0 },
      proxy, { getCurrentTexture: () => color, unconfigure() {} }, pipeline, renderPipeline, cullPipeline, preparePipeline);
    const read = device.createBuffer({ size: 64 * 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      renderer.load(mesh);
      const stats = await new Promise((resolve, reject) => {
        renderer.onStats = resolve; renderer.onError = reject;
        renderer.draw({ axis: 3, offset, angles: [0,0,0,0,0,0],
          yaw: 0.55, pitch: -0.32, distance: 2.8, style: 1, smooth, mode, acceleration });
      });
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer({ texture: color }, { buffer: read, bytesPerRow: 256 }, [64,64]);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const pixels = new Uint8Array(read.getMappedRange()).slice();
      read.unmap();
      return { stats, pixels };
    } finally { renderer.destroy(); read.destroy(); color.destroy(); }
  }
  try {
    for(const mode of [0,1,2]) for (const offset of [0, 1]) for (const smooth of [false,true]) {
      const whole = await draw(1 << 20, offset, smooth, mode, false);
      const batches = await draw(1536, offset, smooth, mode, true); // 8 cells per batch, crossing shared faces
      assert.equal(batches.stats.triangles, whole.stats.triangles);
      assert.equal(batches.stats.coplanar, whole.stats.coplanar);
      assert.deepEqual(batches.pixels, whole.pixels);
    }
  } finally { model.free(); }
});

function te4WithNormals(kind, vertexNormals, cellNormals) {
  const positions = [[0,0,0,-1], [0,1,0,1], [0,0,1,1], [0,-1,-1,1]];
  const bytes = new Uint8Array(12 + 64 + 16 +
    ((kind === 2 || kind === 4) ? 64 : 0) + ((kind === 3 || kind === 4) ? 16 : 0));
  const view = new DataView(bytes.buffer);
  [kind,4,1].forEach((v,i) => view.setUint32(i*4,v,true));
  let at = 12;
  for (const p of positions) for (const value of p) { view.setFloat32(at,value,true); at+=4; }
  for (const value of [1,2,3,4]) { view.setUint32(at,value,true); at+=4; }
  if (kind === 2 || kind === 4) for (const normal of vertexNormals)
    for (const value of normal) { view.setFloat32(at,value,true); at+=4; }
  if (kind === 3 || kind === 4) for (const value of cellNormals) { view.setFloat32(at,value,true); at+=4; }
  return { model: new ViewerModel(bytes,true), positions };
}
function unit(v) {
  const length = Math.hypot(...v);
  return length ? v.map(x => x/length) : v.map(() => 0);
}
function rotatePoint(matrix, point) {
  return [0,1,2,3].map(row => point.reduce((sum,x,col) => sum + matrix[col*4+row]*x,0));
}
for (const kind of [2,3,4]) test(`type ${kind} normals are rotated, interpolated and projected on GPU`, async () => {
  const normals = [[-1,0,0,0],[-1,0.5,0,0],[-1,0,0.4,0],[-1,-0.2,0.3,0]].map(unit);
  const { model, positions } = te4WithNormals(kind,normals,[-1,0,0,0]);
  const angles = [0.31,-0.2,0.4,0.15,0.18,-0.27];
  const matrix = rotation(angles);
  const center = model.center();
  const points = positions.map(p => rotatePoint(matrix,p.map((x,i)=>x-center[i])));
  const offset = points.reduce((sum,p)=>sum+p[3],0)/4;
  const projectedCell = unit(rotatePoint(matrix,[-1,0,0,0]).slice(0,3));
  const expected = [];
  for (let a=0;a<4;a++) for (let b=a+1;b<4;b++) {
    const da = points[a][3]-offset, db = points[b][3]-offset;
    if (da*db >= 0) continue;
    const t = da/(da-db);
    const p = points[a].slice(0,3).map((x,i)=>(x+t*(points[b][i]-x))/model.radius());
    const interpolated = normals[a].map((x,i)=>x+t*(normals[b][i]-x));
    const normal = kind === 3 ? projectedCell : unit(rotatePoint(matrix,interpolated).slice(0,3));
    expected.push({p,normal});
  }
  try {
    const result = await run(model,3,offset,angles);
    assert.ok(result.vertices.length > 0);
    for (let at=0;at<result.vertices.length;at+=8) {
      const match = expected.find(({p})=>p.every((x,i)=>Math.abs(x-result.vertices[at+i])<1e-5));
      assert.ok(match,"intersection matches an original edge");
      for (let i=0;i<3;i++) assert.ok(Math.abs(match.normal[i]-result.vertices[at+4+i])<1e-5);
    }
  } finally { model.free(); }
});

test("zero or plane-parallel projected normals fall back to finite geometric normals", async () => {
  for (const normals of [[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]]) {
    const { model } = te4WithNormals(4,normals,[0,0,0,1]);
    try {
      const result = await run(model,3,0,[0,0,0,0,0,0]);
      for (let at=0;at<result.vertices.length;at+=8) {
        assert.ok(result.vertices.slice(at+4,at+7).every(Number.isFinite));
        assert.ok(Math.abs(Math.hypot(...result.vertices.slice(at+4,at+7))-1)<1e-6);
      }
    } finally { model.free(); }
  }
});

function quadModel(shear, order) {
  const bytes = new Uint8Array(12+64+16), view = new DataView(bytes.buffer);
  [1,4,1].forEach((v,i)=>view.setUint32(i*4,v,true));
  const points=[[-2,0,0,-1],[2,0,0,-1],[shear,-2,0,1],[-shear,2,0,1]];
  let at=12;
  for (const p of points) for (const x of p) {view.setFloat32(at,x,true);at+=4;}
  for (const i of order) {view.setUint32(at,i+1,true);at+=4;}
  return new ViewerModel(bytes,true);
}
function triangleDiagonal(data,stride) {
  const edges=new Map();
  for(let at=0;at<data.length;at+=stride*3) {
    const points=[0,1,2].map(i=>[0,1,2].map(j=>Math.round(data[at+i*stride+j]*1e5)));
    for(let i=0;i<3;i++) {
      const key=[points[i],points[(i+1)%3]].sort((a,b)=>{
        for(let j=0;j<3;j++) if(a[j]!==b[j])return a[j]-b[j];return 0;
      }).flat().join(",");
      edges.set(key,(edges.get(key)??0)+1);
    }
  }
  return [...edges].find(([,count])=>count===2)?.[0];
}
test("GPU quad uses Delaunay and a stable square diagonal across every cell ordering",async()=>{
  const orders=[];
  for(let a=0;a<4;a++)for(let b=0;b<4;b++)for(let c=0;c<4;c++)for(let d=0;d<4;d++) {
    const order=[a,b,c,d]; if(new Set(order).size===4)orders.push(order);
  }
  for(const shear of [0,1])for(const angles of [[0,0,0,0,0,0],[0.37,0.2,0,0.4,0,0]]) {
    let reference;
    for(const order of orders) {
      const model=quadModel(shear,order);
      try {
        const cpu=model.slice(3,0,new Float64Array(angles));
        const result=await run(model,3,0,angles);
        assert.equal(result.values[0],6);
        const diagonal=triangleDiagonal(result.vertices,8);
        assert.equal(diagonal,triangleDiagonal(cpu,6));
        reference??=diagonal;
        assert.equal(diagonal,reference);
      }finally{model.free();}
    }
  }
});

test("animation and pointer updates share a pending frame and draw the latest scene", async () => {
  const {createFrameScheduler}=await import("../src/frame.ts");
  const frames=new Map();let id=0,cancelled=0,scene={offset:0,yaw:0},result;
  const scheduler=createFrameScheduler(compute=>{result={...scene,compute};},
    cb=>{frames.set(++id,cb);return id;}, key=>{cancelled++;frames.delete(key);});
  scene={...scene,offset:0.1};scheduler.schedule(true);
  scene={...scene,yaw:0.7};scheduler.schedule(false);
  scene={...scene,offset:0.2};scheduler.schedule(true);
  assert.equal(frames.size,1);assert.equal(cancelled,0);
  [...frames.values()][0](16);
  assert.deepEqual(result,{offset:0.2,yaw:0.7,compute:true});
});
test("camera events during an animation frame cannot starve an already scheduled draw",async()=>{
  const {createFrameScheduler}=await import("../src/frame.ts");
  let queue=[],nextId=0,draws=0;
  const request=cb=>{const id=++nextId;queue.push({id,cb});return id;};
  const scheduler=createFrameScheduler(()=>{draws++;},request,id=>{queue=queue.filter(x=>x.id!==id);});
  function tick(time){scheduler.schedule(true);request(tick);}
  request(tick);scheduler.schedule(false); // animation callback precedes pending camera draw
  for(let frame=1;frame<=10;frame++){
    const ready=queue;queue=[];
    for(const {cb} of ready)cb(frame*16);
    scheduler.schedule(false); // pointer move before the next refresh
  }
  assert.equal(draws,10);
});
test("frame cleanup cancels pending work and resets the compute flag",async()=>{
  const {createFrameScheduler}=await import("../src/frame.ts");
  const frames=new Map();let id=0,result;
  const scheduler=createFrameScheduler(compute=>{result=compute;},
    cb=>{frames.set(++id,cb);return id;},id=>{frames.delete(id);});
  scheduler.schedule(true);scheduler.cancel();assert.equal(frames.size,0);
  scheduler.schedule(false);[...frames.values()][0](16);assert.equal(result,false);
});

test("a busy GPU accepts only the newest pending camera and section state",async()=>{
  const {Renderer}=await import("../src/gpu.ts");
  globalThis.ResizeObserver=class{observe(){}disconnect(){}};
  globalThis.devicePixelRatio=1;
  const module=device.createShaderModule({code:await readFile(new URL("../public/shaders/render.wgsl",import.meta.url),"utf8")});
  const render=await device.createRenderPipelineAsync({layout:"auto",
    vertex:{module,entryPoint:"vs"},fragment:{module,entryPoint:"fs",targets:[{format:"rgba8unorm"}]},
    primitive:{topology:"triangle-list",cullMode:"none"},
    depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
  const gates=[];let submissions=0,camera,params;
  const queue=new Proxy(device.queue,{get(target,key){
    if(key==="onSubmittedWorkDone")return()=>new Promise(resolve=>gates.push(resolve));
    if(key==="submit")return(commands)=>{submissions++;target.submit(commands);};
    if(key==="writeBuffer")return(...args)=>{
      const [buffer,,bytes]=args;
      if(buffer.size===32)camera=Array.from(new Float32Array(bytes.buffer,bytes.byteOffset,8));
      if(buffer.size===112)params=Array.from(new Float32Array(bytes));
      target.writeBuffer(...args);
    };
    const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const proxy=new Proxy(device,{get(target,key){
    if(key==="queue")return queue;if(key==="destroy")return()=>{};
    const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const color=device.createTexture({size:[64,64],format:"rgba8unorm",usage:GPUTextureUsage.RENDER_ATTACHMENT});
  const renderer=new Renderer({clientWidth:64,clientHeight:64,width:0,height:0},proxy,
    {getCurrentTexture:()=>color,unconfigure(){}},pipeline,render);
  const model=ViewerModel.sample(0);
  try{
    renderer.load({positions:model.positions(),indices:model.gpu_cells(),center:model.center(),radius:model.radius(),
      vertexNormals:model.vertex_normals(),cellNormals:model.cell_normals()});
    const initial={axis:3,offset:0,angles:[0,0,0,0,0,0],yaw:0,pitch:0,distance:2.8,style:1,smooth:false};
    renderer.draw(initial);
    renderer.draw({...initial,yaw:1,offset:0.1});
    renderer.draw({...initial,yaw:2,offset:0.2});
    assert.equal(submissions,1);
    await device.queue.onSubmittedWorkDone();
    gates.shift()();await new Promise(resolve=>setImmediate(resolve));
    assert.equal(submissions,2);
    assert.equal(camera[1],2);
    assert.ok(Math.abs(params[20]-0.2)<1e-6);
    await device.queue.onSubmittedWorkDone();
    gates.shift()();await new Promise(resolve=>setImmediate(resolve));
    assert.equal(submissions,2,"stats readback must not enqueue an extra draw");
  }finally{renderer.destroy();color.destroy();model.free();for(const resolve of gates)resolve();}
});

function canonicalTriangles(data) {
 const rows=[];
 for(let i=0;i<data.length;i+=24)rows.push([0,8,16].map(j=>[0,1,2].map(k=>data[i+j+k].toFixed(5)).join(',')).sort().join(';'));
 return rows.sort();
}
test('GPU BVH matches full scan for rotations, boundary and empty cuts',async()=>{
 for(let kind=0;kind<3;kind++){
  const model=ViewerModel.sample(kind);
  try{for(let axis=0;axis<4;axis++)for(const angles of [[0,0,0,0,0,0],[.11,-.32,.27,.19,.41,-.13]])for(const offset of [0,model.radius(),.173,model.radius()*2]){
   const full=await run(model,axis,offset,angles);
   const culled=await run(model,axis,offset,angles,{bvh:true});
   assert.equal(culled.values[0],full.values[0]);assert.equal(culled.values[4],full.values[4]);
   assert.deepEqual(canonicalTriangles(culled.vertices),canonicalTriangles(full.vertices));
   assert.ok(culled.values[5]<=model.cell_count());
  }}finally{model.free();}
 }
});
test('GPU BVH reduces sparse 8192-cell mesh to a conservative candidate subset',async(t)=>{
 const count=8192, bytes=new Uint8Array(12+count*80),view=new DataView(bytes.buffer);
 [1,count*4,count].forEach((x,i)=>view.setUint32(i*4,x,true));
 let at=12;
 for(let i=0;i<count;i++)for(const p of [[0,0,0,-.4],[1,0,0,.4],[0,1,0,.4],[0,0,1,.4]])for(let axis=0;axis<4;axis++){
  view.setFloat32(at,p[axis]+(axis===3?(i-count/2)*2:0),true);at+=4;
 }
 for(let i=0;i<count*4;i++){view.setUint32(at,i+1,true);at+=4;}
 const model=new ViewerModel(bytes,true);
 try{
  const offset=-model.center()[3];
  const full=await run(model,3,offset,[0,0,0,0,0,0]);
  const culled=await run(model,3,offset,[0,0,0,0,0,0],{bvh:true});
  assert.ok(full.values[0]>0);assert.deepEqual(canonicalTriangles(culled.vertices),canonicalTriangles(full.vertices));
  assert.ok(culled.values[5]<count/20);
  t.diagnostic(`sparse mesh: ${culled.values[5]}/${count} candidate cells; geometry matches full scan`);
  for(const axis of [3,0]){
   const fullTimes=[],bvhTimes=[];
   for(let repeat=0;repeat<7;repeat++){
    const a=await run(model,axis,axis===3?offset:0,[0,0,0,0,0,0]);
    const b=await run(model,axis,axis===3?offset:0,[0,0,0,0,0,0],{bvh:true});
    assert.equal(a.values[0],b.values[0]);fullTimes.push(a.elapsed);bvhTimes.push(b.elapsed);
   }
   const median=a=>a.sort((x,y)=>x-y)[3].toFixed(3);
   t.diagnostic(`lavapipe compute+copy+readback median (7 runs), axis=${axis}: full ${median(fullTimes)} ms, BVH ${median(bvhTimes)} ms; CPU build/upload excluded`);
  }
 }finally{model.free();}
});
test('parallel and perspective projection match unique-face CPU reference',async()=>{
 for(let kind=0;kind<3;kind++){
  const model=ViewerModel.sample(kind);
  try{for(let axis=0;axis<4;axis++)for(const mode of [1,2]){
   const angles=[.11,-.32,.27,.19,.41,-.13],matrix=rotation(angles),center=model.center(),radius=model.radius();
   const positions=model.positions(),cells=model.gpu_cells(),expected=[];
   for(let id=0;id<model.cell_count();id++){
    const points=[];
    for(let corner=0;corner<4;corner++){
     const p=rotatePoint(matrix,Array.from(positions.subarray(cells[id*8+corner]*4,cells[id*8+corner]*4+4), (v,j)=>v-center[j]));
     points.push(p.filter((_,j)=>j!==axis).map(v=>v/radius*(mode===2?3*radius/(3*radius-p[axis]):1)));
    }
    for(let omitted=0;omitted<4;omitted++)if(cells[id*8+4]&(1<<omitted)){
     const face=points.filter((_,j)=>j!==omitted);const a=face[1].map((v,j)=>v-face[0][j]),b=face[2].map((v,j)=>v-face[0][j]);
     if(Math.hypot(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])<=1e-12)continue;
     face.forEach(p=>expected.push(...p,1,0,0,1,0));
    }
   }
   const result=await run(model,axis,999,angles,{mode});
   assert.equal(result.values[0],expected.length/8);
   assert.ok(Math.abs(area(result.vertices,8)-area(expected,8))<1e-4);
   // Every projected vertex must agree with the CPU formula, independent of atomic output order.
   for(let i=0;i<result.vertices.length;i+=8)assert.ok(expected.some((_,j)=>j%8===0&&Math.hypot(...[0,1,2].map(k=>expected[j+k]-result.vertices[i+k]))<2e-6));
   assert.ok(result.vertices.every(Number.isFinite));
  }}finally{model.free();}
 }
});
