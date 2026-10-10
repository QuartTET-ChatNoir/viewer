"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Renderer, type Scene, type Stats } from "../src/gpu";
import { loadCore, type Model } from "../src/model";
import { createFrameScheduler } from "../src/frame";
const INITIAL: Scene = {
  axis: 3,
  offset: 0,
  angles: [0, 0, 0, 0, 0, 0],
  yaw: 0.55,
  pitch: -0.32,
  distance: 2.8,
  style: 1,
  smooth: false,
  mode: 0,
  projectionDistance: 3,
};
const AXES = ["X", "Y", "Z", "W"];
const PLANES = ["XY", "XZ", "XW", "YZ", "YW", "ZW"];
export default function Page() {
  const canvas = useRef<HTMLCanvasElement>(null),
    renderer = useRef<Renderer | null>(null),
    model = useRef<Model | null>(null),
    input = useRef<HTMLInputElement>(null);
  const sceneRef = useRef<Scene>(INITIAL),
    radiusRef = useRef(1),
    durationRef = useRef(30);
  const [scene, setScene] = useState<Scene>(INITIAL),
    [ready, setReady] = useState(false),
    [error, setError] = useState(""),
    [name, setName] = useState("Tesseract / 超立方体"),
    [counts, setCounts] = useState([0, 0]),
    [radius, setRadius] = useState(1),
    [stats, setStats] = useState<Stats>({
      triangles: 0,
      coplanar: 0,
      milliseconds: 0,
    }),
    [oneBased, setOneBased] = useState(true),
    [playing, setPlaying] = useState(false),
    [duration, setDuration] = useState(30),
    [busy, setBusy] = useState(false);
  const redraw = useMemo(() => createFrameScheduler((compute) => {
    renderer.current?.draw(sceneRef.current, compute);
  }), []);
  const change = useCallback((patch: Partial<Scene>, compute = true) => {
    const next = { ...sceneRef.current, ...patch };
    sceneRef.current = next;
    setScene(next);
    redraw.schedule(compute);
  }, [redraw]);
  const install = useCallback(
    (next: Model, title: string) => {
      try {
        renderer.current!.load({
          positions: next.positions(),
          indices: next.gpu_cells(),
          vertexNormals: next.vertex_normals(),
          cellNormals: next.cell_normals(),
          center: next.center(),
          radius: next.radius(),
        });
      } catch (e) {
        next.free();
        throw e;
      }
      model.current?.free();
      model.current = next;
      const r = Math.max(next.radius(), 1e-30);
      radiusRef.current = r;
      setRadius(r);
      setName(title);
      setCounts([next.vertex_count(), next.cell_count()]);
      setStats({ triangles: 0, coplanar: 0, milliseconds: 0 });
      setError("");
      change({ ...INITIAL, angles: [...INITIAL.angles] });
    },
    [change],
  );
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      let gpu: Renderer | undefined;
      try {
        const core = await loadCore();
        if (cancelled) return;
        gpu = await Renderer.create(canvas.current!);
        if (cancelled) {
          gpu.destroy();
          return;
        }
        renderer.current = gpu;
        gpu.onStats = setStats;
        gpu.onError = setError;
        install(core.ViewerModel.sample(0), "Tesseract / 超立方体");
        setReady(true);
      } catch (e) {
        gpu?.destroy();
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
      redraw.cancel();
      renderer.current?.destroy();
      renderer.current = null;
      model.current?.free();
      model.current = null;
    };
  }, [install, redraw]);
  useEffect(() => {
    if (!playing) return;
    let id = 0,
      previous = 0;
    const tick = (now: number) => {
      if (previous) {
        const r = radiusRef.current;
        // Pause elapsed time while hidden; preserve overshoot at the loop boundary.
        const elapsed = document.hidden ? 0 : now - previous;
        const distance = elapsed / (durationRef.current * 1000) * 2 * r;
        const offset = ((sceneRef.current.offset + r + distance) % (2 * r)) - r;
        change({ offset });
      }
      previous = now;
      id = requestAnimationFrame(tick);
    };
    const resetClock = () => { previous = 0; };
    document.addEventListener("visibilitychange", resetClock);
    id = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(id);
      document.removeEventListener("visibilitychange", resetClock);
    };
  }, [playing, change]);
  async function sample(kind: number) {
    setPlaying(false);
    try {
      const core = await loadCore();
      install(
        core.ViewerModel.sample(kind),
        ["Tesseract / 超立方体", "16-cell / 正十六胞体", "5-cell / 正五胞体"][
          kind
        ],
      );
    } catch (e) {
      setError(String(e));
    }
  }
  async function readFile(file: File) {
    if (!ready) return;
    setBusy(true);
    setPlaying(false);
    try {
      const core = await loadCore();
      const bytes = new Uint8Array(await file.arrayBuffer());
      install(new core.ViewerModel(bytes, oneBased), file.name);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const drag = useRef<{ x: number; y: number; pointer: number } | null>(null);
  return (
    <main>
      <header>
        <div className="brand">
          <span className="mark">Q</span>
          <div>
            <h1>
              QuartTET <span>ChatNoir</span>
            </h1>
            <p>FOUR DIMENSIONS. ONE SECTION.</p>
          </div>
        </div>
        <div className="engine">
          <i className={ready ? "active" : ""} />
          {ready ? "WebGPU compute" : "Initializing"}
        </div>
      </header>
      <div className="workspace">
        <aside>
          <section>
            <div className="section-title">
              <span>01</span>
              <h2>モデル</h2>
            </div>
            <button
              className="open"
              disabled={!ready || busy}
              onClick={() => input.current?.click()}
            >
              {busy ? "読み込み中…" : "TE4 ファイルを開く"}
              <span>↗</span>
            </button>
            <input
              ref={input}
              type="file"
              accept=".te4"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void readFile(file);
                e.target.value = "";
              }}
            />
            <label className="check">
              <input
                type="checkbox"
                checked={!oneBased}
                onChange={(e) => setOneBased(!e.target.checked)}
              />
              互換用：頂点番号が 0 始まりのファイル
            </label>
            <div className="samples">
              {["超立方体", "正十六胞体", "正五胞体"].map((s, i) => (
                <button
                  key={s}
                  disabled={!ready || busy}
                  onClick={() => void sample(i)}
                >
                  {s}
                </button>
              ))}
            </div>
            <p className="note">
              type 1–4 · little-endian
              <br />
              ファイルはブラウザ内で処理されます。
            </p>
          </section>
          <section>
            <div className="section-title">
              <span>02</span>
              <h2>切断・投影</h2>
            </div>
            <div className="segmented" aria-label="描画モード">
              {["超平面切断", "投影"].map((label,i)=>(
                <button key={label} disabled={!ready} aria-pressed={i===0?scene.mode===0:scene.mode!==0}
                  onClick={()=>{setPlaying(false);change({mode:i,style:i===1?2:1,smooth:false});}}>
                  {label}
                </button>
              ))}
            </div>
            {scene.mode!==0 && <>
              <div className="segmented" aria-label="4D投影方式">
                {["平行投影", "透視投影"].map((label,i)=>(
                  <button key={label} disabled={!ready} aria-pressed={scene.mode===i+1}
                    onClick={()=>change({mode:i+1})}>{label}</button>
                ))}
              </div>
              {scene.mode===2 && <label>4D視点の距離：{(scene.projectionDistance??3).toFixed(2)} R
                <input type="range" min={1.25} max={8} step={0.05} disabled={!ready} value={scene.projectionDistance}
                  onChange={e=>change({projectionDistance:Number(e.target.value)})}/>
              </label>}
              <p className="note">選択軸を奥行きとして4Dから3Dへ投影します。</p>
            </>}
            <div className="segmented" aria-label={scene.mode===0?"切断軸":"投影軸"}>
              {AXES.map((a, i) => (
                <button
                  key={a}
                  aria-pressed={scene.axis === i}
                  disabled={!ready}
                  onClick={() => {
                    setPlaying(false);
                    change({ axis: i, offset: 0 });
                  }}
                >
                  {a}
                </button>
              ))}
            </div>
            {scene.mode===0 && <>
            <div className="label-row">
              <label htmlFor="offset">{AXES[scene.axis]} =</label>
              <input
                id="offset"
                type="number"
                step={radius / 100}
                min={-radius}
                max={radius}
                value={Number(scene.offset.toPrecision(6))}
                disabled={!ready}
                onChange={(e) => {
                  const v = e.target.valueAsNumber;
                  if (Number.isFinite(v))
                    change({ offset: Math.max(-radius, Math.min(radius, v)) });
                }}
              />
            </div>
            <input
              aria-label="断面の位置"
              type="range"
              min={-1}
              max={1}
              step={0.001}
              value={scene.offset / radius}
              disabled={!ready}
              onChange={(e) =>
                change({ offset: Number(e.target.value) * radius })
              }
            />
            <div className="range-labels">
              <span>−R</span>
              <span>中心</span>
              <span>+R</span>
            </div>
            <button
              className="play"
              disabled={!ready}
              aria-pressed={playing}
              onClick={() => setPlaying(!playing)}
            >
              {playing ? "Ⅱ  停止" : "▷  断面を再生"}
            </button>
            <div className="label-row">
              <label htmlFor="duration">1周の時間（秒）</label>
              <input id="duration" type="number" min={1} max={3600} step={1}
                value={duration} disabled={!ready}
                onChange={(e) => {
                  const value = e.target.valueAsNumber;
                  if (Number.isFinite(value) && value >= 1 && value <= 3600) {
                    durationRef.current = value;
                    setDuration(value);
                  }
                }} />
            </div>
            <p className="note">−R → +R を {duration} 秒で移動します。<br />長くするとゆっくり再生します。</p>
            <p className="note">
              モデル中心を原点とする座標です。
              <br />R はモデルの外接半径。
            </p>
            </>}
          </section>
          <section>
            <div className="section-title">
              <span>03</span>
              <h2>4次元回転</h2>
              <button
                className="text-button"
                disabled={!ready}
                onClick={() => change({ angles: [0, 0, 0, 0, 0, 0] })}
              >
                リセット
              </button>
            </div>
            {PLANES.map((p, i) => (
              <label className="rotation" key={p}>
                <span>{p}</span>
                <input
                  type="range"
                  min={-180}
                  max={180}
                  step={1}
                  value={Math.round((scene.angles[i] * 180) / Math.PI)}
                  disabled={!ready}
                  onChange={(e) => {
                    const angles = [...scene.angles];
                    angles[i] = (Number(e.target.value) * Math.PI) / 180;
                    change({ angles });
                  }}
                />
                <output>
                  {Math.round((scene.angles[i] * 180) / Math.PI)}°
                </output>
              </label>
            ))}
          </section>
          <section>
            <div className="section-title">
              <span>04</span>
              <h2>表示</h2>
            </div>
            <div className="segmented">
              {["面", "面＋辺", "辺"].map((s, i) => (
                <button
                  key={s}
                  disabled={!ready}
                  aria-pressed={scene.style === i}
                  onClick={() => change({ style: i }, false)}
                >
                  {s}
                </button>
              ))}
            </div>
            <div className="segmented" aria-label="陰影">
              {["フラット", "滑らか"].map((label, i) => (
                <button key={label} disabled={!ready || scene.mode!==0}
                  aria-pressed={scene.smooth === (i === 1)}
                  onClick={() => change({ smooth: i === 1 })}>
                  {label}
                </button>
              ))}
            </div>
            <div className="actions">
              <button
                disabled={!ready}
                onClick={() =>
                  change(
                    {
                      yaw: INITIAL.yaw,
                      pitch: INITIAL.pitch,
                      distance: INITIAL.distance,
                    },
                    false,
                  )
                }
              >
                カメラを戻す
              </button>
            </div>
          </section>
        </aside>
        <div
          className="viewport"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) void readFile(file);
          }}
        >
          <div className="view-top">
            <div>
              <span className="eyebrow">{scene.mode===0?"HYPERPLANE SECTION":"4D PROJECTION"}</span>
              <h2>{name}</h2>
            </div>
            <span className="axis-badge">
              {AXES[scene.axis]}
              <small>{scene.mode===0?"定値断面":"投影軸"}</small>
            </span>
          </div>
          <canvas
            ref={canvas}
            aria-label={scene.mode===0?"4次元モデルの3次元断面":"4次元モデルの3次元投影"}
            onPointerDown={(e) => {
              drag.current = {
                x: e.clientX,
                y: e.clientY,
                pointer: e.pointerId,
              };
              e.currentTarget.setPointerCapture(e.pointerId);
            }}
            onPointerMove={(e) => {
              const d = drag.current;
              if (!d || d.pointer !== e.pointerId) return;
              change(
                {
                  yaw: sceneRef.current.yaw + (e.clientX - d.x) * 0.007,
                  pitch: Math.max(
                    -1.5,
                    Math.min(
                      1.5,
                      sceneRef.current.pitch + (e.clientY - d.y) * 0.007,
                    ),
                  ),
                },
                false,
              );
              drag.current = {
                x: e.clientX,
                y: e.clientY,
                pointer: e.pointerId,
              };
            }}
            onPointerUp={() => {
              drag.current = null;
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
            onWheel={(e) =>
              change(
                {
                  distance: Math.max(
                    1.2,
                    Math.min(
                      7,
                      sceneRef.current.distance * Math.exp(e.deltaY * 0.001),
                    ),
                  ),
                },
                false,
              )
            }
          />
          {!ready && !error && (
            <div className="center-message">
              Rust / WASM と WebGPU を準備中…
            </div>
          )}
          {ready && stats.triangles === 0 && (
            <div className="center-message">
              {scene.mode===0?"この位置には面の断面がありません":"この投影には描画可能な面がありません"}
            </div>
          )}
          {error && (
            <div className="error" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="エラーを閉じる">
                ×
              </button>
            </div>
          )}
          {stats.coplanar > 0 && (
            <div className="warning">
              {stats.coplanar.toLocaleString()}{" "}
              胞が切断面と一致しています。体積を持つため面として出力していません。断面を少し移動してください。
            </div>
          )}
          <div className="view-bottom">
            <span>ドラッグで回転 · ホイールでズーム · ファイルをドロップ</span>
            <button
              disabled={!ready}
              onClick={() => {
                setPlaying(false);
                change({ ...INITIAL, angles: [...INITIAL.angles] });
              }}
            >
              全てリセット
            </button>
          </div>
        </div>
      </div>
      <footer>
        <div>
          <span className="stat">
            <b>{counts[0].toLocaleString()}</b> vertices
          </span>
          <span className="stat">
            <b>{counts[1].toLocaleString()}</b> tetrahedra
          </span>
          <span className="stat">
            <b>{stats.triangles.toLocaleString()}</b> triangles
          </span>
        </div>
        <span className="pipeline">GPU SECTION → INDIRECT DRAW</span>
      </footer>
    </main>
  );
}
