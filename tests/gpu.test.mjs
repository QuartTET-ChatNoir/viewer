import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { create, globals } from "webgpu";
import init, { ViewerModel } from "../public/wasm/quarttet_core.js";
Object.assign(globalThis, globals);
let gpu, device, pipeline;
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
async function run(model, axis, offset, angles) {
  const resources = [];
  const buffer = (size, usage) => {
    const b = device.createBuffer({ size, usage });
    resources.push(b);
    return b;
  };
  try {
    const positions = model.positions(),
      indices = model.gpu_cells(),
      count = model.cell_count();
    const input = (data) => {
      const b = buffer(
        data.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      device.queue.writeBuffer(b, 0, data);
      return b;
    };
    const output = buffer(
      count * 192,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const counters = buffer(
      20,
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
    f.set([offset, axis, model.radius() * 1e-6, model.radius()], 20);
    new Uint32Array(raw).set([count, 0, 0, 0], 24);
    device.queue.writeBuffer(uniform, 0, raw);
    device.queue.writeBuffer(counters, 0, new Uint32Array([0, 1, 0, 0, 0]));
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        input(positions),
        input(indices),
        output,
        counters,
        uniform,
      ].map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const read = buffer(
      count * 192 + 32,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    encoder.copyBufferToBuffer(counters, 0, read, 0, 20);
    encoder.copyBufferToBuffer(output, 0, read, 32, count * 192);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const mapped = read.getMappedRange();
    const values = new Uint32Array(mapped, 0, 5).slice();
    const vertices = new Float32Array(mapped, 32, values[0] * 8).slice();
    read.unmap();
    return { values, vertices };
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
