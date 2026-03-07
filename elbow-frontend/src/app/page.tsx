"use client";

import { useState, useRef, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── 型定義 ────────────────────────────────────────────────────────────────────
interface LandmarkPoint {
  x: number; y: number; x_pct: number; y_pct: number;
}
interface ElbowAngles {
  carrying_angle: number | null;
  flexion:        number | null;
  pronation_sup:  number;
  ps_label:       string;
  varus_valgus:   number;
  vv_label:       string;
}
interface QAInfo {
  view_type: string; score: number; status: string;
  message: string; color: string; symmetry_ratio: number;
  positioning_advice: string; inference_engine?: string;
  keypoint_confidences?: number[];
}
interface Landmarks {
  humerus_shaft: LandmarkPoint; condyle_center: LandmarkPoint;
  lateral_epicondyle: LandmarkPoint; medial_epicondyle: LandmarkPoint;
  forearm_shaft: LandmarkPoint; forearm_ext: LandmarkPoint;
  angles: ElbowAngles; qa: QAInfo;
}
interface SecondOpinion {
  carrying_angle: number | null;
  flexion:        number | null;
  pronation_sup:  number;
  varus_valgus:   number;
  model:          string;
}
interface EdgeValidation {
  edge_angle:    number | null;
  agreement_deg: number | null;
  confidence:    "high" | "medium" | "low";
  edge_lines:    number;
  note:          string;
}
interface AnalyzeResponse {
  success: boolean;
  landmarks: Landmarks;
  edge_validation: EdgeValidation | null;
  second_opinion: SecondOpinion | null;
  image_size: { width: number; height: number };
}
interface GradCamResponse {
  success: boolean;
  heatmap_overlay: string;
  raw_heatmap: string;
  predicted_angles: Record<string, number>;
  target: string;
  engine_used: string;
  note: string;
}

// ─── 定数 ──────────────────────────────────────────────────────────────────────
const LANDMARK_COLORS: Record<string, string> = {
  humerus_shaft:      "#3b82f6",
  condyle_center:     "#f97316",
  lateral_epicondyle: "#a855f7",
  medial_epicondyle:  "#ec4899",
  forearm_shaft:      "#22c55e",
};
const GRADCAM_TARGETS = [
  { key: "all",         label: "総合" },
  { key: "carrying",    label: "外反角" },
  { key: "flexion",     label: "屈曲角" },
  { key: "pronation",   label: "回内外" },
  { key: "varus_valgus",label: "内反外反" },
];

// ─── QA バッジ ─────────────────────────────────────────────────────────────────
function QABadge({ qa }: { qa: QAInfo }) {
  const colorMap: Record<string, string> = {
    green:  "bg-green-900 text-green-300 border-green-700",
    yellow: "bg-yellow-900 text-yellow-300 border-yellow-700",
    red:    "bg-red-900 text-red-300 border-red-700",
  };
  const barColor: Record<string, string> = {
    green: "bg-green-500", yellow: "bg-yellow-500", red: "bg-red-500",
  };
  return (
    <div className={`rounded-lg border p-4 ${colorMap[qa.color] || colorMap.green}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-sm">{qa.status}</span>
        <span className="font-mono text-lg font-bold">{qa.score}/100</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
        <div className={`h-2 rounded-full transition-all ${barColor[qa.color]}`} style={{ width: `${qa.score}%` }} />
      </div>
      <p className="text-xs mb-1">{qa.message}</p>
      {qa.positioning_advice && <p className="text-xs mt-2 opacity-80">{qa.positioning_advice}</p>}
      <div className="flex gap-3 mt-2 text-xs opacity-70">
        <span>View: {qa.view_type}</span>
        {qa.inference_engine && <span>Engine: {qa.inference_engine}</span>}
      </div>
    </div>
  );
}

// ─── 角度カード ────────────────────────────────────────────────────────────────
function AngleCard({ title, primary, secondary, unit, label, normalRange, description }: {
  title: string; primary: number | null; secondary?: number | null;
  unit: string; label?: string; normalRange?: string; description?: string;
}) {
  const isNA = primary === null || primary === undefined;
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-700 p-4">
      <p className="text-xs text-gray-400 mb-1">{title}</p>
      {/* プライマリ値（YOLOv8 / Classical CV） */}
      <div className="flex items-baseline gap-1">
        {isNA ? (
          <span className="text-2xl font-bold text-gray-600 font-mono">N/A</span>
        ) : (
          <>
            <span className="text-3xl font-bold text-white font-mono">{primary!.toFixed(1)}</span>
            <span className="text-gray-400 text-sm">{unit}</span>
          </>
        )}
      </div>
      {/* セカンドオピニオン（ConvNeXt） */}
      {secondary !== undefined && (
        <div className="mt-1 flex items-center gap-1 text-xs text-purple-400">
          <span className="opacity-60">ConvNeXt:</span>
          {secondary === null ? (
            <span className="opacity-40">N/A</span>
          ) : (
            <span className="font-mono font-semibold">{secondary.toFixed(1)}{unit}</span>
          )}
        </div>
      )}
      {label && !isNA && <p className="text-xs text-blue-400 mt-1">{label}</p>}
      {normalRange && !isNA && <p className="text-xs text-gray-500 mt-1">正常: {normalRange}</p>}
      {description && <p className="text-xs text-gray-500 mt-1">{isNA ? "この像では測定不可" : description}</p>}
    </div>
  );
}

// ─── ランドマークオーバーレイ ──────────────────────────────────────────────────
function LandmarkOverlay({ landmarks, imageWidth, imageHeight }: {
  landmarks: Landmarks; imageWidth: number; imageHeight: number;
}) {
  const points = [
    { key: "humerus_shaft",      pt: landmarks.humerus_shaft },
    { key: "condyle_center",     pt: landmarks.condyle_center },
    { key: "lateral_epicondyle", pt: landmarks.lateral_epicondyle },
    { key: "medial_epicondyle",  pt: landmarks.medial_epicondyle },
    { key: "forearm_shaft",      pt: landmarks.forearm_shaft },
  ];
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none"
         viewBox={`0 0 ${imageWidth} ${imageHeight}`} preserveAspectRatio="none">
      <line x1={landmarks.humerus_shaft.x} y1={landmarks.humerus_shaft.y}
            x2={landmarks.condyle_center.x} y2={landmarks.condyle_center.y}
            stroke="#3b82f6" strokeWidth="2" strokeDasharray="6,3" opacity="0.8" />
      <line x1={landmarks.condyle_center.x} y1={landmarks.condyle_center.y}
            x2={landmarks.forearm_ext.x} y2={landmarks.forearm_ext.y}
            stroke="#22c55e" strokeWidth="2" strokeDasharray="6,3" opacity="0.8" />
      <line x1={landmarks.lateral_epicondyle.x} y1={landmarks.lateral_epicondyle.y}
            x2={landmarks.medial_epicondyle.x}   y2={landmarks.medial_epicondyle.y}
            stroke="#f97316" strokeWidth="2" opacity="0.7" />
      {points.map(({ key, pt }) => (
        <circle key={key} cx={pt.x} cy={pt.y} r="7"
          fill={LANDMARK_COLORS[key] || "#fff"} stroke="white" strokeWidth="2" opacity="0.9" />
      ))}
    </svg>
  );
}

// ─── メインコンポーネント ─────────────────────────────────────────────────────
export default function ElbowVisionPage() {
  const [imageData,      setImageData]      = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(null);
  const [gradCamResult,  setGradCamResult]  = useState<GradCamResponse | null>(null);
  const [activeTab,      setActiveTab]      = useState<"analysis" | "gradcam">("analysis");
  const [gradCamTarget,  setGradCamTarget]  = useState("all");
  const [isAnalyzing,    setIsAnalyzing]    = useState(false);
  const [isGradCam,      setIsGradCam]      = useState(false);
  const [dragOver,       setDragOver]       = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [currentFile,    setCurrentFile]    = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const analyzeFile = useCallback(async (file: File) => {
    setError(null); setIsAnalyzing(true); setGradCamResult(null);
    try {
      const uploadForm = new FormData();
      uploadForm.append("file", file);
      const uploadRes = await fetch(`${API_URL}/api/upload`, { method: "POST", body: uploadForm });
      if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
      const uploadData = await uploadRes.json();
      setImageData(uploadData.image_data);

      const analyzeForm = new FormData();
      analyzeForm.append("file", file);
      const analyzeRes = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: analyzeForm });
      if (!analyzeRes.ok) throw new Error(`Analysis failed: ${analyzeRes.status}`);
      const data: AnalyzeResponse = await analyzeRes.json();
      setAnalysisResult(data);
      setCurrentFile(file);
      setActiveTab("analysis");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleFile = (file: File) => {
    const allowed = [".dcm", ".dicom", ".png", ".jpg", ".jpeg"];
    if (!allowed.some(a => file.name.toLowerCase().endsWith(a))) {
      setError("対応形式: DICOM (.dcm), PNG, JPEG"); return;
    }
    analyzeFile(file);
  };

  const runGradCam = async (target: string = gradCamTarget) => {
    if (!currentFile) return;
    setIsGradCam(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", currentFile);
      const res = await fetch(`${API_URL}/api/gradcam?target=${target}`, { method: "POST", body: form });
      const data = await res.json();
      // 503 = モデル未訓練。success:false のまま結果にセットして黄色バナーで表示
      setGradCamResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "GradCAM error");
    } finally {
      setIsGradCam(false);
    }
  };

  const so = analysisResult?.second_opinion;

  return (
    <div className="min-h-screen bg-gray-950">
      {/* ヘッダー */}
      <header className="border-b border-gray-800 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white"><span className="text-blue-400">Elbow</span>Vision</h1>
            <p className="text-xs text-gray-400">AI 肘関節角度解析システム v1.0</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block"></span>
            YOLOv8-Pose + ConvNeXt + FastAPI
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ── 左カラム: 画像 ── */}
          <div className="space-y-4">
            {!imageData && (
              <div
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
                  dragOver ? "border-blue-500 bg-blue-950/30" : "border-gray-700 hover:border-gray-500"
                }`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
              >
                <div className="text-5xl mb-4">🦴</div>
                <p className="text-lg font-medium text-gray-300">肘X線画像をドロップ</p>
                <p className="text-sm text-gray-500 mt-1">または クリックして選択</p>
                <p className="text-xs text-gray-600 mt-3">対応形式: DICOM (.dcm) / PNG / JPEG</p>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept=".dcm,.dicom,.png,.jpg,.jpeg" className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />

            {imageData && (
              <div className="relative rounded-xl overflow-hidden bg-black border border-gray-700">
                <img
                  src={activeTab === "gradcam" && gradCamResult ? gradCamResult.heatmap_overlay : imageData}
                  alt="肘X線" className="w-full object-contain"
                />
                {activeTab === "analysis" && analysisResult && (
                  <LandmarkOverlay landmarks={analysisResult.landmarks}
                    imageWidth={analysisResult.image_size.width} imageHeight={analysisResult.image_size.height} />
                )}
                <button onClick={() => { setImageData(null); setAnalysisResult(null); setGradCamResult(null); setCurrentFile(null); }}
                  className="absolute top-2 right-2 bg-black/60 hover:bg-black text-white text-xs px-3 py-1 rounded-full">
                  再アップロード
                </button>
              </div>
            )}

            {analysisResult && (
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(LANDMARK_COLORS).map(([key, color]) => (
                  <div key={key} className="flex items-center gap-2 text-xs text-gray-400">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
                    <span>{{ humerus_shaft: "上腕骨", condyle_center: "顆部",
                      lateral_epicondyle: "外側上顆", medial_epicondyle: "内側上顆",
                      forearm_shaft: "前腕" }[key]}</span>
                  </div>
                ))}
              </div>
            )}

            {error && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">{error}</div>
            )}
            {isAnalyzing && (
              <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-4 text-center">
                <div className="animate-spin text-2xl mb-2">⚙️</div>
                <p className="text-blue-300 text-sm">AI解析中...</p>
              </div>
            )}
          </div>

          {/* ── 右カラム: 結果 ── */}
          <div className="space-y-4">
            {analysisResult ? (
              <>
                {/* タブ */}
                <div className="flex gap-2">
                  <button onClick={() => setActiveTab("analysis")}
                    className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
                      activeTab === "analysis" ? "bg-blue-700 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>
                    ランドマーク解析
                  </button>
                  <button onClick={() => { setActiveTab("gradcam"); if (!gradCamResult) runGradCam(); }}
                    disabled={isGradCam}
                    className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                      activeTab === "gradcam" ? "bg-purple-700 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>
                    {isGradCam ? "生成中..." : "Grad-CAM XAI"}
                  </button>
                </div>

                {/* ── 解析タブ ── */}
                {activeTab === "analysis" && (
                  <>
                    <QABadge qa={analysisResult.landmarks.qa} />

                    {/* エッジバリデーション */}
                    {analysisResult.edge_validation && (() => {
                      const ev = analysisResult.edge_validation!;
                      const colorMap = {
                        high:   "bg-green-900/40 border-green-700 text-green-300",
                        medium: "bg-yellow-900/40 border-yellow-700 text-yellow-300",
                        low:    "bg-red-900/40 border-red-700 text-red-300",
                      };
                      const icon = { high: "✓", medium: "△", low: "✗" };
                      return (
                        <div className={`rounded-lg border px-3 py-2 text-xs flex items-start gap-2 ${colorMap[ev.confidence]}`}>
                          <span className="font-bold text-base leading-none mt-0.5">{icon[ev.confidence]}</span>
                          <div>
                            <span className="font-semibold">エッジ検証 </span>
                            {ev.edge_angle !== null
                              ? <span className="font-mono">{ev.edge_angle}°（差 {ev.agreement_deg}°）</span>
                              : <span>検出失敗</span>}
                            <span className="ml-2 opacity-70">{ev.note}</span>
                          </div>
                        </div>
                      );
                    })()}

                    {/* ConvNeXt モデル説明バナー */}
                    {so && (
                      <div className="bg-purple-950/40 border border-purple-800 rounded-lg px-3 py-2 text-xs text-purple-300 flex items-center gap-2">
                        <span className="text-purple-400 font-bold">ConvNeXt</span>
                        セカンドオピニオン有効 — 各カードに紫色で表示
                      </div>
                    )}
                    {!so && (
                      <div className="bg-gray-900/40 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-500 flex items-center gap-2">
                        <span>ConvNeXt:</span> モデル未訓練（セカンドオピニオン無効）
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-3">
                      <AngleCard title="外反角（Carrying Angle）" unit="°"
                        primary={analysisResult.landmarks.angles.carrying_angle}
                        secondary={so?.carrying_angle}
                        normalRange="5〜15°" description="AP像: 上腕骨軸と前腕骨軸の角度" />
                      <AngleCard title="屈曲角（Flexion）" unit="°"
                        primary={analysisResult.landmarks.angles.flexion}
                        secondary={so?.flexion}
                        normalRange="0〜150°" description="側面像: 伸展0° / 完全屈曲150°" />
                      <AngleCard title="回内外（Pronation/Supination）" unit="°"
                        primary={analysisResult.landmarks.angles.pronation_sup}
                        secondary={so?.pronation_sup}
                        label={analysisResult.landmarks.angles.ps_label}
                        normalRange="回内80° / 回外85°" />
                      <AngleCard title="内反外反（Varus/Valgus）" unit="°"
                        primary={analysisResult.landmarks.angles.varus_valgus}
                        secondary={so?.varus_valgus}
                        label={analysisResult.landmarks.angles.vv_label}
                        normalRange="正常: 0〜2°" />
                    </div>

                    {/* ランドマーク座標テーブル */}
                    <div className="bg-gray-900 rounded-xl border border-gray-700 p-4">
                      <h3 className="text-sm font-medium text-gray-300 mb-3">ランドマーク座標</h3>
                      <table className="w-full text-xs text-gray-400">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="text-left pb-2">点</th>
                            <th className="text-right pb-2">X</th>
                            <th className="text-right pb-2">Y</th>
                            <th className="text-right pb-2">X%</th>
                            <th className="text-right pb-2">Y%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(["humerus_shaft","condyle_center","lateral_epicondyle","medial_epicondyle","forearm_shaft"] as const).map(key => {
                            const pt = analysisResult.landmarks[key];
                            return (
                              <tr key={key} className="border-b border-gray-800">
                                <td className="py-1.5" style={{ color: LANDMARK_COLORS[key] }}>
                                  {{ humerus_shaft: "上腕骨", condyle_center: "顆部中心",
                                     lateral_epicondyle: "外側上顆", medial_epicondyle: "内側上顆",
                                     forearm_shaft: "前腕骨" }[key]}
                                </td>
                                <td className="text-right py-1.5 font-mono">{pt.x}</td>
                                <td className="text-right py-1.5 font-mono">{pt.y}</td>
                                <td className="text-right py-1.5 font-mono">{pt.x_pct.toFixed(1)}</td>
                                <td className="text-right py-1.5 font-mono">{pt.y_pct.toFixed(1)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {/* ── Grad-CAM タブ ── */}
                {activeTab === "gradcam" && (
                  <div className="space-y-4">
                    <div className="flex gap-2 flex-wrap">
                      {GRADCAM_TARGETS.map(t => (
                        <button key={t.key}
                          onClick={() => { setGradCamTarget(t.key); setGradCamResult(null); }}
                          className={`py-1.5 px-3 rounded-lg text-sm transition-colors ${
                            gradCamTarget === t.key ? "bg-purple-700 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}>
                          {t.label}
                        </button>
                      ))}
                      <button onClick={() => runGradCam(gradCamTarget)} disabled={isGradCam || !currentFile}
                        className="py-1.5 px-4 rounded-lg text-sm bg-purple-700 hover:bg-purple-600 disabled:opacity-50 transition-colors">
                        {isGradCam ? "生成中..." : "生成"}
                      </button>
                    </div>

                    {gradCamResult?.success ? (
                      <>
                        <div className="bg-gray-900 rounded-xl border border-gray-700 p-4">
                          <p className="text-xs text-gray-400 mb-3">{gradCamResult.note}</p>
                          <div className="grid grid-cols-2 gap-3 text-xs">
                            {Object.entries(gradCamResult.predicted_angles).map(([k, v]) => (
                              <div key={k} className="bg-gray-800 rounded-lg p-2">
                                <p className="text-gray-500">{k}</p>
                                <p className="text-white font-mono font-bold">{v.toFixed(1)}°</p>
                              </div>
                            ))}
                          </div>
                        </div>
                        {gradCamResult.raw_heatmap && (
                          <div>
                            <p className="text-xs text-gray-400 mb-2">ヒートマップのみ</p>
                            <img src={gradCamResult.raw_heatmap} alt="GradCAM heatmap"
                              className="w-full rounded-xl border border-gray-700" />
                          </div>
                        )}
                      </>
                    ) : gradCamResult && !gradCamResult.success ? (
                      <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 text-yellow-300 text-sm">
                        <p className="font-medium mb-1">ConvNeXt モデル未訓練</p>
                        <p className="text-xs opacity-80">elbow_convnext_best.pth を訓練・配置後に利用可能になります。</p>
                      </div>
                    ) : !isGradCam ? (
                      <div className="text-center text-gray-500 py-10">
                        <p className="text-4xl mb-3">🔬</p>
                        <p className="text-sm">「生成」ボタンでGrad-CAM XAIを実行</p>
                        <p className="text-xs mt-1 text-gray-600">ConvNeXt-Smallが「どこを見て判断したか」を可視化</p>
                      </div>
                    ) : null}
                  </div>
                )}
              </>
            ) : (
              !isAnalyzing && (
                <div className="text-center text-gray-600 py-16">
                  <p className="text-5xl mb-4">🦴</p>
                  <p className="text-lg text-gray-500">肘X線画像をアップロードしてください</p>
                  <p className="text-sm mt-2">DICOM / PNG / JPEG 対応</p>
                  <div className="mt-6 text-left bg-gray-900 rounded-xl p-4 border border-gray-800 text-xs text-gray-500 space-y-1">
                    <p className="text-gray-400 font-medium mb-2">測定項目 + 2モデル比較</p>
                    <p>• 外反角（Carrying Angle）— AP像 / 正常 5〜15°</p>
                    <p>• 屈曲角（Flexion）— 側面像 / 0〜150°</p>
                    <p>• 回内外（Pronation/Supination）</p>
                    <p>• 内反外反（Varus/Valgus）</p>
                    <p className="mt-2 text-purple-500">• ConvNeXt セカンドオピニオン</p>
                    <p className="text-purple-500">• Grad-CAM XAI 可視化</p>
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
