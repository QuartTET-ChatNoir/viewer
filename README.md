# QuartTET-ChatNoir viewer

4次元四面体メッシュを3次元超平面で切断するブラウザビューア。Next.js + React、Rust/WASM による te4 読込、WebGPU compute による断面抽出と間接描画を使います。

## 初版の機能

- 最小 te4 type 1 のファイル選択 / ドロップ、1始まりの頂点番号の明示選択、0始まり te4 への保存
- X/Y/Z/W 定値断面、断面位置のスライダー・数値入力・ループ再生
- XY/XZ/XW/YZ/YW/ZW 回転、3D カメラ回転とズーム
- 面 / 面＋辺 / 辺 の表示
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

1. Rust/WASM で te4 を読み込み、長さ・インデックス・有限座標を検査。共有面の所有胞を一度だけ決定。
2. 元の4D頂点・胞・共有面ビットマスクを GPU に転送。
3. core の `shaders/section.wgsl` で各胞の4D回転、超平面交差判定、辺上の交点補間、三角形化。
4. atomic 頂点カウンタで出力を詰め、`drawIndirect` でそのまま描画。

通常表示では Rust の CPU 断面処理を呼びません。GPU → CPU の読戻しは20 bytesの統計だけです。カメラと表示方式だけの変更では断面を再計算しません。

## 依存と検証

2026-10-07 JST に最新版を確認して初版を構築しました。

| 依存                         | 初版のバージョン |
| ---------------------------- | ---------------- |
| Node.js                      | 26.10.0          |
| npm                          | 12.2.0           |
| Rust stable                  | 1.99.0           |
| wasm-bindgen                 | 0.2.129          |
| Next.js                      | 16.3.8           |
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

- type 1 / little-endian の最小 te4。法線付き type 2–4、テクスチャ、周期性は今後の拡張。
- 任意超平面はモデル側の6平面回転と軸選択で指定。切断位置はモデル中心基準。
- 共面四面体の交差は体積になるため表示せず、該当胞数を案内。
- GPU 計算は f32、外接半径 × 1e-6 を交点許容誤差に使用。
- ファイル上限64 MiB、頂点数・胞数各200万、さらに使用 GPU の storage buffer 上限を検査。
- 出力容量は最大192 bytes / 胞を確保するため、大規模データのメモリ効率は改善余地あり。
- 初回 te4 検査と GPU 用準備はメインスレッド。巨大ファイルでは一時的に UI が停止し得ます。
