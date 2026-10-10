import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "QuartTET · ChatNoir",
  description: "4次元四面体メッシュの WebGPU 断面ビューア",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
