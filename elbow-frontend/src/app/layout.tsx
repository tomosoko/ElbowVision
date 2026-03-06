import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ElbowVision — AI 肘関節角度解析",
  description: "YOLOv8-Poseによる肘X線画像からの6DoF角度推定システム",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body className="min-h-screen bg-gray-950 text-gray-100">
        {children}
      </body>
    </html>
  );
}
