import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, copyFileSync } from "node:fs";
import { resolve } from "node:path";
const core = resolve(process.env.QUARTTET_CORE_PATH ?? "../core");
if (!existsSync(resolve(core, "Cargo.toml")))
  throw new Error(
    "Clone QuartTET-ChatNoir/core next to viewer, or set QUARTTET_CORE_PATH.",
  );
mkdirSync("public/wasm", { recursive: true });
const mode =
  spawnSync("wasm-bindgen", ["--version"]).status === 0
    ? "no-install"
    : "normal";
execFileSync(
  process.env.WASM_PACK_BIN ?? process.execPath,
  [
    ...(process.env.WASM_PACK_BIN
      ? []
      : [resolve("node_modules/wasm-pack/run.js")]),
    "build",
    core,
    "--target",
    "web",
    "--out-dir",
    resolve("public/wasm"),
    "--out-name",
    "quarttet_core",
    "--mode",
    mode,
  ],
  { stdio: "inherit" },
);

mkdirSync("public/shaders", { recursive: true });
copyFileSync(
  resolve(core, "shaders/section.wgsl"),
  resolve("public/shaders/section.wgsl"),
);
