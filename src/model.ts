export type Model = {
  free(): void;
  positions(): Float32Array;
  indices(): Uint32Array;
  gpu_cells(): Uint32Array;
  vertex_normals(): Float32Array;
  cell_normals(): Float32Array;
  center(): Float64Array;
  radius(): number;
  vertex_count(): number;
  cell_count(): number;
};
export type Core = {
  ViewerModel: {
    new (bytes: Uint8Array, oneBased: boolean): Model;
    sample(kind: number): Model;
  };
};
let loaded: Promise<Core> | undefined;
export function loadCore(): Promise<Core> {
  if (!loaded)
    loaded = (async () => {
      const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
      // Keep generated JS as a native module served next to its WASM binary.
      const url = `${base}/wasm/quarttet_core.js`;
      const core = await import(/* webpackIgnore: true */ url);
      await core.default();
      return core as Core;
    })().catch((error) => {
      loaded = undefined;
      throw error;
    });
  return loaded;
}
