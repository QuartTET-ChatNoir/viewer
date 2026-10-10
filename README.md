# QuartTET-ChatNoir viewer

4次元四面体メッシュを3次元超平面で切断するブラウザビューア。Next.js + React、Rust/WASM による te4 読込、WebGPU compute による断面抽出と間接描画を使います。

## 初版の機能

- te4 type 1–4 のファイル選択 / ドロップ、標準の1始まり頂点番号、旧0始まりファイルの互換設定
- X/Y/Z/W 定値断面、断面位置のスライダー・数値入力・ループ再生（1周1〜3600秒、初期値30秒）
- XY/XZ/XW/YZ/YW/ZW 回転、3D カメラ回転とズーム
- 面 / 面＋辺 / 辺 の表示、フラット / 滑らかな陰影
- 超立方体、正十六胞体、正五胞体の組込みサンプル
- 頂点・胞・断面三角形数の表示、共面胞・不正入力・GPU 上限の案内

ファイルはブラウザ内で処理します。サーバーへのモデル送信はありません。WebGPU が利用できる最新版ブラウザの HTTPS または localhost 環境が必要です。

## セットアップ

最新版 Node.js と npm を用意し、core と viewer を隣に配置します。配布 ZIP の `core/` と `viewer/` に各リポジトリのソースが入っています。導入手順は同梱の `README-SETUP.md` を参照してください。

```sh
git clone https://github.com/QuartTET-ChatNoir/core.git core
git clone https://github.com/QuartTET-ChatNoir/viewer.git viewer
cd viewer
rustup update stable
rustup target add wasm32-unknown-unknown
npm ci
npm run wasm
npm run dev
```

<http://localhost:3000> を開きます。core の場所は `QUARTTET_CORE_PATH` で変更できます。`WASM_PACK_BIN` は wasm-pack の実行パスを上書きします。

`public/wasm` と `public/shaders/section.wgsl` はビルド生成物です。core のソースから生成し、viewer の Git には重複保存しません。CI は `core-revision.txt` に指定された core を取得します。配布版は `main` を指定しているため、初期導入でコミット番号を書き換える必要はありません。

## 構成

1. Rust/WASM で te4 を読み込み、長さ・インデックス・有限座標・法線を検査。不足する4D法線を一度生成し、共有面の所有胞を決定。
2. 4D頂点・胞・共有面ビットマスク・4D頂点法線・4D胞法線をバッチ単位で GPU に転送。
3. core の `shaders/section.wgsl` で各胞の4D回転、超平面交差判定、辺上の交点・法線補間、三角形化、法線の3D射影。
4. atomic 頂点カウンタで出力を詰め、`drawIndirect` でそのまま描画。

通常表示では Rust の CPU 断面処理を呼びません。GPU → CPU の読戻しは1バッチあたり20 bytesの統計だけです。入力をバッチごとに再配置し、最大32 MiBの断面出力バッファを使い回して計算・描画します。カメラと表示方式の変更でも断面を再計算します。

## 依存と検証

2026-10-10 JST に Node.js / npm / Next.js / React / TypeScript の最新版を再確認しました。

| 依存                         | 初版のバージョン |
| ---------------------------- | ---------------- |
| Node.js                      | 26.11.1          |
| npm                          | 12.2.0           |
| Rust stable                  | 1.99.0           |
| wasm-bindgen                 | 0.2.129          |
| Next.js                      | 16.4.0           |
| React / React DOM            | 19.3.0           |
| TypeScript                   | 7.0.2            |
| wasm-pack                    | 0.15.0           |
| Dawn node-webgpu（テスト用） | 0.6.2            |

npm / Cargo のロックファイルを追跡しています。TypeScript 7 の DOM 宣言と WebGPU の型定義の重複は `skipLibCheck` で避け、アプリ本体は strict 型チェックを行います。

```sh
npm run wasm
npm run typecheck
npm run test:gpu
npm run build
```

GPU テストには WebGPU/Vulkan adapter が必要です。Linux のソフトウェア実行には Mesa lavapipe を使用できます。CI は `mesa-vulkan-drivers` を用意します。

初版の検証: Rust 9 tests、GPU 14 tests、strict 型チェック、Next.js 静的ビルド成功。GPU テストは3形状 × 4軸 × 回転有無 × 3位置を CPU 参照断面の頂点数・交点・面積と照合し、共面胞と描画結果も検証します。検証はソフトウェア Vulkan。実機ブラウザの操作確認・性能測定は未実施です。

`npm run build` は `out/` を生成します。GitHub Pages などのサブディレクトリに置く場合はビルド前に `NEXT_PUBLIC_BASE_PATH=/viewer` を設定してください。静的成果物には WASM とシェーダーも含まれます。

## GitHub Pages

最初に viewer の **Settings → Pages → Build and deployment → Source** を **GitHub Actions** に設定してください。

その後は `main` への push で、テスト・ビルドの成功後に自動公開します。手動実行も Actions の WebGPU viewer → Run workflow から可能です。pull request は検証のみで公開しません。

通常の公開先: <https://quarttet-chatnoir.github.io/viewer/>

Pages が返す base path を Next.js、WASM、シェーダーの URL に反映するため、`/viewer/` 配下でもアセットを読み込めます。core の初回 push を先に完了してください。

## 初版の範囲

- type 1–4 / little-endian の te4。付属4D法線を保持。不足する胞・頂点法線は読込時にRustで一度生成して保持し、GPUで回転・交点補間・3Dへの射影。「フラット / 滑らか」を切り替えられます。テクスチャ、周期性は未対応。
- 任意超平面はモデル側の6平面回転と軸選択で指定。切断位置はモデル中心基準。
- 共面四面体の交差は体積になるため表示せず、該当胞数を案内。
- GPU 計算は f32、外接半径 × 1e-6 を交点許容誤差に使用。
- 固定のファイル容量・頂点数・胞数上限は設けません。ブラウザ / WASM / CPU / GPU の実メモリ容量には依存します。
- 出力容量は1バッチ最大192 bytes / 胞、全バッチで最大32 MiBの出力バッファを共有。入力バッチはGPUに保持するため、GPU入力メモリはモデル規模に比例します。
- 初回 te4 検査と GPU 用準備はメインスレッド。巨大ファイルでは一時的に UI が停止し得ます。

## 2026-10-10 修正の検証

Rust 12 tests（type 1–4、64 MiB超・200万頂点超、体積重み付き法線）、GPU 19 tests（法線の回転・補間・射影、ゼロ射影時のフォールバック、分割前後のピクセル・統計一致）、型チェック、Pages用静的ビルドをLinuxで確認。GPUはMesa lavapipe。実機ブラウザと実際の大規模TE4は未検証です。

法線の初回生成・共有面準備はCPUで行い、断面アニメーション中はGPUで処理します。欠損法線の生成には胞の頂点順序の整合を前提とし、角を保つ場合はフラット表示を使います。4D法線の保持には概算16 ×（頂点数＋胞数）bytesを追加で使います。

## 断面の対角線・TE4番号の修正

TE4は1始まりを標準として読み込み、内部は0始まりです。0始まりの旧ファイルだけ互換用チェックを付けます。断面四角形の対角線は局所ドロネー条件で選び、正方形など同等の場合は元モデルの頂点番号から決定して固定します。各胞の境界線は保持します。GPUバッチ内の頂点番号に依存しないよう、元番号を別バッファで渡します。

## 再生中のカメラ操作

断面更新とカメラ操作は1つの描画予約へまとめます。新しい入力が来ても予約を取り消さず、実行時に最新の断面・カメラ状態を使用します。GPU処理中は次の状態を1つだけ保持し、完了後に最新状態を描画します。統計の読戻し完了から余分な描画を発行しません。

2026-10-11 JST: 24 tests（GPU形状検証、予約の競合、連続カメラ入力時の描画継続、GPU待機中の最新状態への集約）、型チェック、Pages用静的ビルドをLinux / Mesa lavapipeで確認。実機ブラウザでのマウス操作確認は未実施です。

候補抽出の木は未追加です。現状は全胞をGPUで走査し、非交差胞を早期終了します。大規模・低交差率のモデルでは4D AABB/BVHのGPU候補抽出と全走査の比較計測が必要です。
