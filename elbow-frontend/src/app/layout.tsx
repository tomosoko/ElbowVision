import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ElbowVision -- AI 肘関節角度解析",
  description:
    "YOLOv8-Pose + ConvNeXt-Small による肘X線画像からの多角度推定・ポジショニング補正ガイドシステム",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja" className="dark">
      <body className="min-h-screen bg-[#030712] text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
