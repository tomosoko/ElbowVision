"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// --- 型定義 ---
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
  radial_head?: LandmarkPoint; olecranon?: LandmarkPoint;
  angles: ElbowAngles; qa: QAInfo;
}
interface SecondOpinion {
  rotation_error_deg: number | null;
  flexion_deg:        number | null;
  model:              string;
}
interface EdgeValidation {
  edge_angle:    number | null;
  agreement_deg: number | null;
  confidence:    "high" | "medium" | "low";
  edge_lines:    number;
  note:          string;
}
interface PositioningCorrection {
  view_type:          string;
  epic_separation_px: number;
  epic_ratio:         number;
  rotation_error:     number;
  rotation_level:     "good" | "minor" | "major";
  rotation_advice:    string;
  flexion_deg:        number | null;
  flexion_level:      "good" | "minor" | "major" | null;
  flexion_advice:     string | null;
  overall_level:      "good" | "minor" | "major";
  correction_needed:  boolean;
}
interface AnalyzeResponse {
  success: boolean;
  landmarks: Landmarks;
  edge_validation: EdgeValidation | null;
  positioning_correction: PositioningCorrection | null;
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
interface HealthStatus {
  status: string;
  version: string;
  engines: {
    yolo_pose: boolean;
    convnext_xai: boolean;
    gradcam_xai: boolean;
    classical_cv: boolean;
  };
}
interface HistoryEntry {
  id: string;
  timestamp: string;
  fileName: string;
  imageData: string;
  result: AnalyzeResponse;
}

// --- 定数 ---
const LANDMARK_COLORS: Record<string, string> = {
  humerus_shaft:      "#3b82f6",
  condyle_center:     "#f97316",
  lateral_epicondyle: "#a855f7",
  medial_epicondyle:  "#ec4899",
  forearm_shaft:      "#22c55e",
  radial_head:        "#facc15",
  olecranon:          "#f87171",
};
const LANDMARK_LABELS: Record<string, string> = {
  humerus_shaft: "上腕骨", condyle_center: "顆部中心",
  lateral_epicondyle: "外側上顆", medial_epicondyle: "内側上顆",
  forearm_shaft: "前腕骨", radial_head: "橈骨頭", olecranon: "肘頭",
};
// --- 左腕用テキスト反転 ---
function flipPSText(text: string): string {
  return text
    .replace(/回外 \(Supination\)/g, "回内 (Pronation)")
    .replace(/回内 \(Pronation\)/g, "回外_TMP")
    .replace(/回外_TMP/g, "回外 (Supination)")
    .replace(/回外/g, "回内_TMP")
    .replace(/回内(?!_TMP)/g, "回外")
    .replace(/回内_TMP/g, "回内");
}
function flipAdvice(text: string): string {
  const t = text
    .replace(/回外してください/g, "回内_TMPしてください")
    .replace(/回内しています/g, "回外_TMPしています")
    .replace(/回外位/g, "回内_TMP位")
    .replace(/回内_TMP/g, "回内")
    .replace(/回外_TMP/g, "回外")
    .replace(/手のひらが上を向/g, "手のひらが下を向")
    .replace(/手のひらをさらに上に向ける/g, "手のひらをさらに下に向ける")
    .replace(/手のひらが完全に上を向く/g, "手のひらが完全に下を向く");
  return t;
}

const GRADCAM_TARGETS = [
  { key: "all",      label: "総合" },
  { key: "rotation", label: "回旋ズレ(AP)" },
  { key: "flexion",  label: "屈曲角(LAT)" },
];

// --- LocalStorage ヘルパー ---
function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem("elbowvision_history");
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}
function saveHistory(entries: HistoryEntry[]) {
  try {
    // 最大20件、画像データが大きいので制限
    const trimmed = entries.slice(0, 20);
    localStorage.setItem("elbowvision_history", JSON.stringify(trimmed));
  } catch { /* quota exceeded */ }
}

// --- 角度ゲージ SVG ---
function AngleGauge({ value, max, color, size = 80 }: {
  value: number; max: number; color: string; size?: number;
}) {
  const radius = (size - 8) / 2;
  const circumference = Math.PI * radius; // 半円
  const pct = Math.min(Math.abs(value) / max, 1);
  const offset = circumference * (1 - pct);
  return (
    <svg width={size} height={size / 2 + 8} viewBox={`0 0 ${size} ${size / 2 + 8}`} className="block">
      <path
        d={`M 4 ${size / 2 + 4} A ${radius} ${radius} 0 0 1 ${size - 4} ${size / 2 + 4}`}
        fill="none" stroke="#1f2937" strokeWidth="6" strokeLinecap="round"
      />
      <path
        d={`M 4 ${size / 2 + 4} A ${radius} ${radius} 0 0 1 ${size - 4} ${size / 2 + 4}`}
        fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={offset}
        className="transition-all duration-700"
      />
      <text x={size / 2} y={size / 2} textAnchor="middle" fill="white"
            fontSize="14" fontWeight="bold" fontFamily="monospace">
        {value.toFixed(1)}°
      </text>
    </svg>
  );
}

// --- QA バッジ ---
function QABadge({ qa }: { qa: QAInfo }) {
  const colorMap: Record<string, string> = {
    green:  "border-emerald-600/60 bg-emerald-950/40",
    yellow: "border-amber-600/60 bg-amber-950/40",
    red:    "border-red-600/60 bg-red-950/40",
  };
  const barColor: Record<string, string> = {
    green: "#22c55e", yellow: "#eab308", red: "#ef4444",
  };
  const statusIcon: Record<string, string> = {
    green: "GOOD", yellow: "FAIR", red: "POOR",
  };
  return (
    <div className={`rounded-xl border p-4 ${colorMap[qa.color] || colorMap.green}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-bold text-sm text-gray-200">品質スコア</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
            qa.color === "green" ? "bg-emerald-800/60 text-emerald-300" :
            qa.color === "yellow" ? "bg-amber-800/60 text-amber-300" :
            "bg-red-800/60 text-red-300"
          }`}>{statusIcon[qa.color] || "GOOD"}</span>
        </div>
        <span className="font-mono text-2xl font-bold text-white">{qa.score}</span>
      </div>
      {/* カラーバー */}
      <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden mb-3 relative">
        <div className="absolute inset-0 flex">
          <div className="h-full bg-emerald-600/30" style={{ width: "60%" }} />
          <div className="h-full bg-amber-600/30" style={{ width: "20%" }} />
          <div className="h-full bg-red-600/30" style={{ width: "20%" }} />
        </div>
        <div
          className="h-full rounded-full relative z-10 transition-all duration-700"
          style={{ width: `${qa.score}%`, backgroundColor: barColor[qa.color] }}
        />
      </div>
      <p className="text-xs text-gray-300 mb-1">{qa.message}</p>
      {qa.positioning_advice && (
        <p className="text-xs mt-2 text-gray-400 bg-gray-900/50 rounded-lg px-3 py-2">{qa.positioning_advice}</p>
      )}
      <div className="flex gap-4 mt-3 text-xs text-gray-500">
        <span>View: <span className="text-gray-300 font-medium">{qa.view_type}</span></span>
        {qa.inference_engine && <span>Engine: <span className="text-gray-300 font-medium">{qa.inference_engine}</span></span>}
        {qa.symmetry_ratio !== undefined && (
          <span>Symmetry: <span className="text-gray-300 font-mono">{qa.symmetry_ratio.toFixed(2)}</span></span>
        )}
      </div>
    </div>
  );
}

// --- 角度カード ---
function AngleCard({ title, primary, secondary, unit, label, normalRange, description }: {
  title: string; primary: number | null; secondary?: number | null;
  unit: string; label?: string; normalRange?: string; description?: string;
}) {
  const isNA = primary === null || primary === undefined;
  const gaugeMax = title.includes("外反角") ? 30 : title.includes("屈曲角") ? 150 : 40;
  const gaugeColor = isNA ? "#374151" :
    title.includes("外反角") ? "#3b82f6" :
    title.includes("屈曲角") ? "#22c55e" :
    title.includes("回内外") ? "#f97316" : "#a855f7";

  return (
    <div className="bg-gradient-to-br from-gray-900 to-gray-900/80 rounded-xl border border-gray-700/60 p-2 hover:border-gray-600 transition-colors">
      <p className="text-xs text-gray-400 mb-2 font-medium">{title}</p>
      {!isNA ? (
        <div className="flex items-center gap-3">
          <AngleGauge value={Math.abs(primary!)} max={gaugeMax} color={gaugeColor} size={56} />
          <div>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-white font-mono">{primary!.toFixed(1)}</span>
              <span className="text-gray-400 text-sm">{unit}</span>
            </div>
            {label && <p className="text-xs text-blue-400 mt-0.5">{label}</p>}
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <div className="w-[72px] h-[44px] flex items-center justify-center">
            <span className="text-2xl font-bold text-gray-600 font-mono">N/A</span>
          </div>
        </div>
      )}
      {secondary !== undefined && (
        <div className="mt-2 flex items-center gap-1 text-xs text-purple-400 bg-purple-950/30 rounded-lg px-2 py-1">
          <span className="opacity-60">ConvNeXt:</span>
          {secondary === null ? (
            <span className="opacity-40">N/A</span>
          ) : (
            <span className="font-mono font-semibold">{secondary.toFixed(1)}{unit}</span>
          )}
        </div>
      )}
      {normalRange && !isNA && <p className="text-xs text-gray-500 mt-2">正常範囲: {normalRange}</p>}
      {description && <p className="text-xs text-gray-600 mt-1">{isNA ? "この像では測定不可" : description}</p>}
    </div>
  );
}

// --- ポジショニングカード ---
function PositioningCard({ pc }: { pc: PositioningCorrection }) {
  const bgMap = {
    good:  "border-emerald-600/50 bg-gradient-to-br from-emerald-950/50 to-emerald-950/20",
    minor: "border-amber-600/50 bg-gradient-to-br from-amber-950/50 to-amber-950/20",
    major: "border-red-600/50 bg-gradient-to-br from-red-950/50 to-red-950/20",
  };
  const titleColor = { good: "text-emerald-300", minor: "text-amber-300", major: "text-red-300" };
  const badge = { good: "良好", minor: "要調整", major: "要再撮影" };
  const badgeBg = {
    good:  "bg-emerald-800/60 text-emerald-200",
    minor: "bg-amber-800/60 text-amber-200",
    major: "bg-red-800/60 text-red-200",
  };
  const levelIcon = { good: "check", minor: "warning", major: "error" };
  const levelIconMap: Record<string, string> = { good: "✓", minor: "!", major: "✕" };

  return (
    <div className={`rounded-xl border-2 p-5 ${bgMap[pc.overall_level]}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className={`font-bold text-sm ${titleColor[pc.overall_level]}`}>
          ポジショニングガイダンス
        </h3>
        <span className={`text-xs font-bold px-3 py-1 rounded-full ${badgeBg[pc.overall_level]}`}>
          {badge[pc.overall_level]}
        </span>
      </div>

      {/* 回旋ズレ */}
      <div className={`rounded-xl p-4 mb-3 border ${
        pc.rotation_level === "good" ? "bg-emerald-900/20 border-emerald-800/40" :
        pc.rotation_level === "minor" ? "bg-amber-900/20 border-amber-800/40" : "bg-red-900/20 border-red-800/40"
      }`}>
        <div className="flex items-center gap-2 mb-2">
          <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
            pc.rotation_level === "good" ? "bg-emerald-700 text-emerald-100" :
            pc.rotation_level === "minor" ? "bg-amber-700 text-amber-100" : "bg-red-700 text-red-100"
          }`}>{levelIconMap[pc.rotation_level]}</span>
          <span className="text-sm font-semibold text-gray-200">回旋位置</span>
          {pc.rotation_error > 0 && (
            <span className="ml-auto font-mono text-sm text-white bg-gray-800/80 px-2 py-0.5 rounded">
              {pc.rotation_error.toFixed(0)}° ズレ
            </span>
          )}
        </div>
        <p className="text-xs text-gray-300 leading-relaxed">{flipAdvice(pc.rotation_advice)}</p>
      </div>

      {/* 屈曲角（LAT像のみ） */}
      {pc.flexion_deg !== null && pc.flexion_advice && (
        <div className={`rounded-xl p-4 border ${
          pc.flexion_level === "good" ? "bg-emerald-900/20 border-emerald-800/40" :
          pc.flexion_level === "minor" ? "bg-amber-900/20 border-amber-800/40" : "bg-red-900/20 border-red-800/40"
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              pc.flexion_level === "good" ? "bg-emerald-700 text-emerald-100" :
              pc.flexion_level === "minor" ? "bg-amber-700 text-amber-100" : "bg-red-700 text-red-100"
            }`}>{levelIconMap[pc.flexion_level || "good"]}</span>
            <span className="text-sm font-semibold text-gray-200">屈曲角</span>
            <span className="ml-auto font-mono text-sm text-white bg-gray-800/80 px-2 py-0.5 rounded">
              {pc.flexion_deg}° (目標: 90°)
            </span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{pc.flexion_advice}</p>
        </div>
      )}

      <div className="flex gap-4 mt-3 text-xs text-gray-600">
        <span>外顆間距離: {pc.epic_separation_px}px</span>
        <span>体格比: {pc.epic_ratio}</span>
      </div>
    </div>
  );
}

// --- ランドマークオーバーレイ ---
function LandmarkOverlay({ landmarks, imageWidth, imageHeight }: {
  landmarks: Landmarks; imageWidth: number; imageHeight: number;
}) {
  const corePoints = [
    { key: "humerus_shaft",      pt: landmarks.humerus_shaft },
    { key: "condyle_center",     pt: landmarks.condyle_center },
    { key: "lateral_epicondyle", pt: landmarks.lateral_epicondyle },
    { key: "medial_epicondyle",  pt: landmarks.medial_epicondyle },
    { key: "forearm_shaft",      pt: landmarks.forearm_shaft },
  ];
  const extra: { key: string; pt: LandmarkPoint }[] = [];
  if (landmarks.radial_head) extra.push({ key: "radial_head",  pt: landmarks.radial_head });
  if (landmarks.olecranon)   extra.push({ key: "olecranon",    pt: landmarks.olecranon });
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none"
         viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
      {/* 骨軸ライン */}
      <line x1={landmarks.humerus_shaft.x} y1={landmarks.humerus_shaft.y}
            x2={landmarks.condyle_center.x} y2={landmarks.condyle_center.y}
            stroke="#3b82f6" strokeWidth="1" strokeDasharray="4,2" opacity="0.7" />
      <line x1={landmarks.condyle_center.x} y1={landmarks.condyle_center.y}
            x2={landmarks.forearm_ext.x} y2={landmarks.forearm_ext.y}
            stroke="#22c55e" strokeWidth="1" strokeDasharray="4,2" opacity="0.7" />
      {/* 上顆間ライン（延長） */}
      {(() => {
        const lx = landmarks.lateral_epicondyle.x, ly = landmarks.lateral_epicondyle.y;
        const mx = landmarks.medial_epicondyle.x, my = landmarks.medial_epicondyle.y;
        const dx = lx - mx, dy = ly - my;
        const ext = 1.5;
        return (
          <>
            <line x1={mx - dx * ext} y1={my - dy * ext}
                  x2={lx + dx * ext} y2={ly + dy * ext}
                  stroke="#f97316" strokeWidth="0.5" strokeDasharray="3,3" opacity="0.3" />
            <line x1={lx} y1={ly} x2={mx} y2={my}
                  stroke="#f97316" strokeWidth="0.8" opacity="0.6" />
            {/* 水平基準線 */}
            <line x1={mx - dx * ext} y1={(ly + my) / 2}
                  x2={lx + dx * ext} y2={(ly + my) / 2}
                  stroke="#ffffff" strokeWidth="0.3" strokeDasharray="2,4" opacity="0.25" />
          </>
        );
      })()}
      {/* 橈骨頭ライン */}
      {landmarks.radial_head && (
        <line x1={landmarks.condyle_center.x} y1={landmarks.condyle_center.y}
              x2={landmarks.radial_head.x}     y2={landmarks.radial_head.y}
              stroke="#facc15" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
      )}
      {/* 肘頭ライン */}
      {landmarks.olecranon && (
        <line x1={landmarks.condyle_center.x} y1={landmarks.condyle_center.y}
              x2={landmarks.olecranon.x}       y2={landmarks.olecranon.y}
              stroke="#f87171" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
      )}
      {/* コアポイント */}
      {corePoints.map(({ key, pt }) => (
        <g key={key}>
          <circle cx={pt.x} cy={pt.y} r="3" fill={LANDMARK_COLORS[key] || "#fff"} opacity="0.2" />
          <circle cx={pt.x} cy={pt.y} r="2" fill={LANDMARK_COLORS[key] || "#fff"} stroke="white" strokeWidth="0.5" opacity="0.9" />
        </g>
      ))}
      {/* 追加ポイント */}
      {extra.map(({ key, pt }) => {
        const r = 2;
        const d = `M ${pt.x} ${pt.y - r} L ${pt.x + r} ${pt.y} L ${pt.x} ${pt.y + r} L ${pt.x - r} ${pt.y} Z`;
        return (
          <g key={key}>
            <circle cx={pt.x} cy={pt.y} r="3" fill={LANDMARK_COLORS[key]} opacity="0.15" />
            <path d={d} fill={LANDMARK_COLORS[key]} stroke="white" strokeWidth="0.5" opacity="0.9" />
          </g>
        );
      })}
    </svg>
  );
}

// --- 比較モード ---
function CompareView({ entries, onClose }: { entries: HistoryEntry[]; onClose: () => void }) {
  const [leftIdx, setLeftIdx]   = useState(0);
  const [rightIdx, setRightIdx] = useState(Math.min(1, entries.length - 1));

  if (entries.length < 2) {
    return (
      <div className="bg-gray-900 rounded-xl border border-gray-700 p-8 text-center">
        <p className="text-gray-400">比較には2件以上の履歴が必要です。</p>
        <button onClick={onClose} className="mt-4 text-sm text-blue-400 hover:underline">閉じる</button>
      </div>
    );
  }

  const left  = entries[leftIdx];
  const right = entries[rightIdx];

  const renderSummary = (entry: HistoryEntry) => {
    const a = entry.result.landmarks.angles;
    return (
      <div className="space-y-2 text-sm">
        <img src={entry.imageData} alt="X線" className="w-full rounded-lg border border-gray-700 object-contain max-h-[300px]" />
        <p className="text-xs text-gray-500 truncate">{entry.fileName}</p>
        <p className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleString("ja-JP")}</p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-gray-800 rounded-lg p-2">
            <span className="text-gray-400">外反角</span>
            <span className="block font-mono text-white font-bold">
              {a.carrying_angle !== null ? `${a.carrying_angle.toFixed(1)}°` : "N/A"}
            </span>
          </div>
          <div className="bg-gray-800 rounded-lg p-2">
            <span className="text-gray-400">屈曲角</span>
            <span className="block font-mono text-white font-bold">
              {a.flexion !== null ? `${a.flexion.toFixed(1)}°` : "N/A"}
            </span>
          </div>
          <div className="bg-gray-800 rounded-lg p-2">
            <span className="text-gray-400">回内外</span>
            <span className="block font-mono text-white font-bold">{a.pronation_sup.toFixed(1)}°</span>
          </div>
          <div className="bg-gray-800 rounded-lg p-2">
            <span className="text-gray-400">内反外反</span>
            <span className="block font-mono text-white font-bold">{a.varus_valgus.toFixed(1)}°</span>
          </div>
        </div>
        <div className="text-xs">
          <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
            entry.result.landmarks.qa.color === "green" ? "bg-emerald-800/60 text-emerald-300" :
            entry.result.landmarks.qa.color === "yellow" ? "bg-amber-800/60 text-amber-300" :
            "bg-red-800/60 text-red-300"
          }`}>QA: {entry.result.landmarks.qa.score}/100</span>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-900/95 rounded-xl border border-gray-700 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-gray-200">画像比較モード</h3>
        <button onClick={onClose} className="text-xs text-gray-400 hover:text-white px-3 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors">
          閉じる
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <select
            className="w-full mb-3 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-300"
            value={leftIdx} onChange={e => setLeftIdx(Number(e.target.value))}
          >
            {entries.map((e, i) => (
              <option key={e.id} value={i}>{e.fileName} - {new Date(e.timestamp).toLocaleString("ja-JP")}</option>
            ))}
          </select>
          {renderSummary(left)}
        </div>
        <div>
          <select
            className="w-full mb-3 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-300"
            value={rightIdx} onChange={e => setRightIdx(Number(e.target.value))}
          >
            {entries.map((e, i) => (
              <option key={e.id} value={i}>{e.fileName} - {new Date(e.timestamp).toLocaleString("ja-JP")}</option>
            ))}
          </select>
          {renderSummary(right)}
        </div>
      </div>
    </div>
  );
}

// --- 履歴パネル ---
function HistoryPanel({ entries, onSelect, onClear }: {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
}) {
  if (entries.length === 0) {
    return (
      <div className="text-center text-gray-600 py-6 text-xs">
        解析履歴はまだありません
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-400 font-medium">直近 {entries.length} 件</span>
        <button onClick={onClear}
          className="text-xs text-red-400/70 hover:text-red-400 transition-colors">
          履歴をクリア
        </button>
      </div>
      <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
        {entries.map(entry => (
          <button key={entry.id} onClick={() => onSelect(entry)}
            className="w-full flex items-center gap-3 bg-gray-800/60 hover:bg-gray-800 border border-gray-700/50 hover:border-gray-600 rounded-lg p-2.5 transition-all text-left">
            <img src={entry.imageData} alt="" className="w-10 h-10 rounded object-cover bg-black flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs text-gray-300 truncate">{entry.fileName}</p>
              <p className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleString("ja-JP")}</p>
            </div>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full flex-shrink-0 ${
              entry.result.landmarks.qa.color === "green" ? "bg-emerald-900/60 text-emerald-400" :
              entry.result.landmarks.qa.color === "yellow" ? "bg-amber-900/60 text-amber-400" :
              "bg-red-900/60 text-red-400"
            }`}>{entry.result.landmarks.qa.score}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// メインコンポーネント
// =============================================================================
export default function ElbowVisionPage() {
  const [imageData,      setImageData]      = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeResponse | null>(null);
  const [gradCamResult,  setGradCamResult]  = useState<GradCamResponse | null>(null);
  const [activeTab,      setActiveTab]      = useState<"analysis" | "gradcam" | "history" | "compare">("analysis");
  const [gradCamTarget,  setGradCamTarget]  = useState("all");
  const [isAnalyzing,    setIsAnalyzing]    = useState(false);
  const [isGradCam,      setIsGradCam]      = useState(false);
  const [dragOver,       setDragOver]       = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [currentFile,    setCurrentFile]    = useState<File | null>(null);
  const [health,         setHealth]         = useState<HealthStatus | null>(null);
  const [history,        setHistory]        = useState<HistoryEntry[]>([]);
  const [showLandmarks,  setShowLandmarks]  = useState(true);
  const [gradCamOpacity, setGradCamOpacity] = useState(100);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 初期ロード
  useEffect(() => {
    setHistory(loadHistory());
    fetch(`${API_URL}/api/health`)
      .then(r => r.json())
      .then(d => setHealth(d))
      .catch(() => setHealth(null));
  }, []);

  const addToHistory = useCallback((fileName: string, imgData: string, result: AnalyzeResponse) => {
    const entry: HistoryEntry = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      fileName,
      imageData: imgData,
      result,
    };
    setHistory(prev => {
      const next = [entry, ...prev].slice(0, 20);
      saveHistory(next);
      return next;
    });
  }, []);

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
      addToHistory(file.name, uploadData.image_data, data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setIsAnalyzing(false);
    }
  }, [addToHistory]);

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
      setGradCamResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "GradCAM error");
    } finally {
      setIsGradCam(false);
    }
  };

  const loadFromHistory = (entry: HistoryEntry) => {
    setImageData(entry.imageData);
    setAnalysisResult(entry.result);
    setGradCamResult(null);
    setCurrentFile(null);
    setActiveTab("analysis");
  };

  const resetView = () => {
    setImageData(null);
    setAnalysisResult(null);
    setGradCamResult(null);
    setCurrentFile(null);
    setError(null);
  };

  const so = analysisResult?.second_opinion;

  return (
    <div className="min-h-screen bg-gray-950">
      {/* ===== ヘッダー ===== */}
      <header className="border-b border-gray-800/80 bg-gray-900/90 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 2 L12 8 M12 8 L8 14 M12 8 L16 14 M8 14 L8 22 M16 14 L16 22" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold text-white">
                <span className="text-blue-400">Elbow</span>Vision
              </h1>
              <p className="text-[10px] text-gray-500 hidden sm:block">AI Elbow Joint Angle Analysis System</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {health && (
              <div className="hidden sm:flex items-center gap-2 text-xs text-gray-500 bg-gray-800/50 rounded-lg px-3 py-1.5">
                <span className={`w-2 h-2 rounded-full ${health.status === "ok" ? "bg-emerald-500" : "bg-red-500"}`} />
                <span className="hidden md:inline">
                  YOLO: {health.engines.yolo_pose ? "ON" : "OFF"} |
                  ConvNeXt: {health.engines.convnext_xai ? "ON" : "OFF"} |
                  Grad-CAM: {health.engines.gradcam_xai ? "ON" : "OFF"}
                </span>
                <span className="md:hidden">API {health.version}</span>
              </div>
            )}
            {!health && (
              <div className="flex items-center gap-2 text-xs text-red-400 bg-red-950/30 rounded-lg px-3 py-1.5">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                API 未接続
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-2 sm:py-3">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-3 sm:gap-4">

          {/* ===== 左カラム: 画像 (3/5) ===== */}
          <div className="lg:col-span-3 space-y-2">
            {!imageData && !isAnalyzing && (
              <div
                className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-300 ${
                  dragOver
                    ? "border-blue-400 bg-blue-950/30 scale-[1.01]"
                    : "border-gray-700 hover:border-gray-500 hover:bg-gray-900/50"
                }`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
              >
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-800 flex items-center justify-center">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="1.5" strokeLinecap="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
                  </svg>
                </div>
                <p className="text-base sm:text-lg font-medium text-gray-300">
                  {dragOver ? "ここにドロップ" : "肘X線画像をドロップ"}
                </p>
                <p className="text-sm text-gray-500 mt-1">またはクリックして選択</p>
                <div className="flex items-center justify-center gap-2 mt-4">
                  {[".dcm", ".png", ".jpg"].map(ext => (
                    <span key={ext} className="text-xs bg-gray-800 text-gray-400 px-2 py-1 rounded">{ext}</span>
                  ))}
                </div>
              </div>
            )}
            <input ref={fileInputRef} type="file" accept=".dcm,.dicom,.png,.jpg,.jpeg" className="hidden"
              onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />

            {/* 解析中 */}
            {isAnalyzing && (
              <div className="bg-blue-950/30 border border-blue-800/50 rounded-2xl p-8 text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-900/50 mb-3">
                  <svg className="animate-spin w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.3" />
                    <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                </div>
                <p className="text-blue-300 text-sm font-medium">AI解析中...</p>
                <p className="text-blue-400/50 text-xs mt-1">YOLOv8-Pose + ConvNeXt + Edge Validation</p>
              </div>
            )}

            {/* 画像表示 */}
            {imageData && (
              <div className="relative rounded-2xl overflow-hidden bg-black border border-gray-700/60 group flex items-center justify-center">
                <div className="relative w-full"
                     style={analysisResult ? { aspectRatio: `${analysisResult.image_size.width} / ${analysisResult.image_size.height}` } : undefined}>
                {activeTab === "gradcam" && gradCamResult?.success ? (
                  <div className="relative">
                    <img src={imageData} alt="元画像" className="w-full h-full object-fill" />
                    <img
                      src={gradCamResult.heatmap_overlay}
                      alt="Grad-CAMオーバーレイ"
                      className="absolute inset-0 w-full h-full object-fill transition-opacity duration-300"
                      style={{ opacity: gradCamOpacity / 100 }}
                    />
                  </div>
                ) : (
                  <img src={imageData} alt="肘X線" className="w-full h-full object-fill" />
                )}
                {activeTab === "analysis" && analysisResult && showLandmarks && (
                  <LandmarkOverlay landmarks={analysisResult.landmarks}
                    imageWidth={analysisResult.image_size.width} imageHeight={analysisResult.image_size.height} />
                )}
                </div>
                {/* オーバーレイコントロール */}
                <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  {activeTab === "analysis" && analysisResult && (
                    <button
                      onClick={() => setShowLandmarks(!showLandmarks)}
                      className="bg-black/70 hover:bg-black/90 text-white text-xs px-3 py-1.5 rounded-lg backdrop-blur-sm transition-colors"
                    >
                      {showLandmarks ? "ランドマーク非表示" : "ランドマーク表示"}
                    </button>
                  )}
                  <button onClick={resetView}
                    className="bg-black/70 hover:bg-black/90 text-white text-xs px-3 py-1.5 rounded-lg backdrop-blur-sm transition-colors">
                    再アップロード
                  </button>
                </div>
                {/* Grad-CAM透過度スライダー */}
                {activeTab === "gradcam" && gradCamResult?.success && (
                  <div className="absolute bottom-3 left-3 right-3 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2 flex items-center gap-3">
                    <span className="text-xs text-gray-300 flex-shrink-0">オーバーレイ</span>
                    <input
                      type="range" min="0" max="100" value={gradCamOpacity}
                      onChange={e => setGradCamOpacity(Number(e.target.value))}
                      className="flex-1 h-1 bg-gray-600 rounded-full appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-gray-400 font-mono w-8 text-right">{gradCamOpacity}%</span>
                  </div>
                )}
              </div>
            )}

            {/* ランドマーク凡例 */}
            {analysisResult && activeTab === "analysis" && showLandmarks && (
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                {(Object.entries(LANDMARK_COLORS) as [string, string][]).map(([key, color]) => {
                  if ((key === "radial_head" || key === "olecranon") &&
                      !analysisResult.landmarks[key as "radial_head" | "olecranon"]) return null;
                  return (
                    <div key={key} className="flex items-center gap-2 text-xs text-gray-400">
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color }} />
                      <span>{LANDMARK_LABELS[key] ?? key}</span>
                    </div>
                  );
                })}
              </div>
            )}

            {/* エラー表示 */}
            {error && (
              <div className="bg-red-950/30 border border-red-800/50 rounded-xl p-4 flex items-start gap-3">
                <span className="text-red-400 text-lg flex-shrink-0">!</span>
                <div>
                  <p className="text-red-300 text-sm font-medium">エラー</p>
                  <p className="text-red-400/80 text-xs mt-0.5">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* ===== 右カラム: 結果 (2/5) ===== */}
          <div className="lg:col-span-2 space-y-2 max-h-[calc(100vh-60px)] overflow-y-auto">
            {/* タブナビゲーション */}
            <div className="flex gap-1 bg-gray-900 rounded-xl p-1 border border-gray-800">
              {([
                { key: "analysis", label: "解析結果",  disabled: false },
                { key: "gradcam",  label: "Grad-CAM",  disabled: false },
                { key: "history",  label: `履歴(${history.length})`, disabled: false },
                { key: "compare",  label: "比較",       disabled: history.length < 2 },
              ] as const).map(tab => (
                <button
                  key={tab.key}
                  disabled={tab.disabled}
                  onClick={() => {
                    setActiveTab(tab.key);
                    if (tab.key === "gradcam" && !gradCamResult && currentFile) runGradCam();
                  }}
                  className={`flex-1 py-2 px-2 rounded-lg text-xs font-medium transition-all disabled:opacity-30 ${
                    activeTab === tab.key
                      ? tab.key === "gradcam"
                        ? "bg-purple-700 text-white shadow-sm"
                        : "bg-blue-700 text-white shadow-sm"
                      : "text-gray-400 hover:bg-gray-800 hover:text-gray-300"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* === 解析タブ === */}
            {activeTab === "analysis" && (
              analysisResult ? (
                <div className="space-y-4">
                  {/* ポジショニングガイダンス */}
                  {analysisResult.positioning_correction && (
                    <PositioningCard pc={analysisResult.positioning_correction} />
                  )}

                  <QABadge qa={analysisResult.landmarks.qa} />

                  {/* エッジバリデーション */}
                  {analysisResult.edge_validation && (() => {
                    const ev = analysisResult.edge_validation!;
                    const colorMap = {
                      high:   "border-emerald-700/50 bg-emerald-950/30",
                      medium: "border-amber-700/50 bg-amber-950/30",
                      low:    "border-red-700/50 bg-red-950/30",
                    };
                    const iconMap = { high: "✓", medium: "!", low: "✕" };
                    const iconColorMap = {
                      high:   "bg-emerald-700 text-emerald-100",
                      medium: "bg-amber-700 text-amber-100",
                      low:    "bg-red-700 text-red-100",
                    };
                    return (
                      <div className={`rounded-xl border px-4 py-3 flex items-start gap-3 ${colorMap[ev.confidence]}`}>
                        <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${iconColorMap[ev.confidence]}`}>
                          {iconMap[ev.confidence]}
                        </span>
                        <div className="text-xs">
                          <span className="font-semibold text-gray-200">エッジ検証 </span>
                          {ev.edge_angle !== null
                            ? <span className="font-mono text-gray-300">{ev.edge_angle}° (差 {ev.agreement_deg}°)</span>
                            : <span className="text-gray-400">検出失敗</span>}
                          <p className="text-gray-400 mt-0.5">{ev.note}</p>
                        </div>
                      </div>
                    );
                  })()}

                  {/* ConvNeXt未訓練バナー */}
                  {!so && (
                    <div className="bg-gray-900/50 border border-gray-700/50 rounded-xl px-4 py-3 text-xs text-gray-500 flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-gray-800 flex items-center justify-center text-gray-600 text-[10px]">i</span>
                      ConvNeXt モデル未訓練 -- セカンドオピニオン無効
                    </div>
                  )}

                  {/* 角度カード */}
                  <div className="grid grid-cols-2 gap-2">
                    <AngleCard title="外反角 (Carrying Angle)" unit="°"
                      primary={analysisResult.landmarks.angles.carrying_angle}
                      secondary={so?.rotation_error_deg}
                      normalRange="5 - 15°" description="AP像: 上腕骨軸と前腕骨軸の角度" />
                    <AngleCard title="屈曲角 (Flexion)" unit="°"
                      primary={analysisResult.landmarks.angles.flexion}
                      secondary={so?.flexion_deg}
                      normalRange="0 - 150°" description="側面像: 伸展0° / 完全屈曲150°" />
                    <AngleCard title="回内外 (Pron./Supi.)" unit="°"
                      primary={-analysisResult.landmarks.angles.pronation_sup}
                      label={flipPSText(analysisResult.landmarks.angles.ps_label)}
                      normalRange="回内80° / 回外85°" />
                    <AngleCard title="内反外反 (Varus/Valgus)" unit="°"
                      primary={analysisResult.landmarks.angles.varus_valgus}
                      label={analysisResult.landmarks.angles.vv_label}
                      normalRange="正常: 0 - 2°" />
                  </div>

                  {/* ConvNeXt ポジショニングズレ */}
                  {so && (
                    <div className="bg-gradient-to-br from-purple-950/40 to-purple-950/20 border border-purple-700/50 rounded-xl p-5">
                      <h3 className="text-sm font-semibold text-purple-300 mb-3 flex items-center gap-2">
                        <span className="w-5 h-5 rounded bg-purple-700 flex items-center justify-center text-xs text-purple-200">C</span>
                        ConvNeXt ポジショニングズレ推定
                      </h3>
                      <div className="grid grid-cols-2 gap-3">
                        {so.rotation_error_deg !== null && (
                          <div className="bg-purple-900/20 border border-purple-800/30 rounded-xl p-4">
                            <div className="text-gray-400 text-xs mb-2">回旋ズレ量 (AP)</div>
                            <div className="font-mono text-white text-xl font-bold">
                              {so.rotation_error_deg > 0 ? "+" : ""}{so.rotation_error_deg.toFixed(1)}°
                            </div>
                            <div className={`text-xs mt-2 px-2 py-0.5 rounded-full inline-block ${
                              Math.abs(so.rotation_error_deg) <= 5 ? "bg-emerald-900/50 text-emerald-400" :
                              Math.abs(so.rotation_error_deg) <= 15 ? "bg-amber-900/50 text-amber-400" :
                              "bg-red-900/50 text-red-400"
                            }`}>
                              {Math.abs(so.rotation_error_deg) <= 5 ? "良好" :
                               Math.abs(so.rotation_error_deg) <= 15 ? "軽度ズレ" : "要補正"}
                            </div>
                          </div>
                        )}
                        {so.flexion_deg !== null && (
                          <div className="bg-purple-900/20 border border-purple-800/30 rounded-xl p-4">
                            <div className="text-gray-400 text-xs mb-2">屈曲角 (LAT)</div>
                            <div className="font-mono text-white text-xl font-bold">
                              {so.flexion_deg.toFixed(1)}°
                            </div>
                            <div className={`text-xs mt-2 px-2 py-0.5 rounded-full inline-block ${
                              Math.abs(so.flexion_deg - 90) <= 10 ? "bg-emerald-900/50 text-emerald-400" :
                              Math.abs(so.flexion_deg - 90) <= 20 ? "bg-amber-900/50 text-amber-400" :
                              "bg-red-900/50 text-red-400"
                            }`}>
                              {Math.abs(so.flexion_deg - 90) <= 10 ? "良好 (目標: 90°)" :
                               Math.abs(so.flexion_deg - 90) <= 20 ? "軽度ズレ" : "要補正"}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* ランドマーク座標テーブル */}
                  <details className="bg-gray-900/80 rounded-xl border border-gray-700/50 overflow-hidden">
                    <summary className="px-4 py-3 text-sm font-medium text-gray-300 cursor-pointer hover:bg-gray-800/50 transition-colors">
                      ランドマーク座標
                    </summary>
                    <div className="px-4 pb-4">
                      <table className="w-full text-xs text-gray-400">
                        <thead>
                          <tr className="border-b border-gray-700">
                            <th className="text-left pb-2 font-medium">点</th>
                            <th className="text-right pb-2 font-medium">X</th>
                            <th className="text-right pb-2 font-medium">Y</th>
                            <th className="text-right pb-2 font-medium">X%</th>
                            <th className="text-right pb-2 font-medium">Y%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(["humerus_shaft","condyle_center","lateral_epicondyle","medial_epicondyle","forearm_shaft","radial_head","olecranon"] as const).map(key => {
                            const pt = analysisResult.landmarks[key];
                            if (!pt) return null;
                            return (
                              <tr key={key} className="border-b border-gray-800/50">
                                <td className="py-1.5 font-medium" style={{ color: LANDMARK_COLORS[key] }}>
                                  {LANDMARK_LABELS[key] ?? key}
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
                  </details>

                  {/* キーポイント信頼度 */}
                  {analysisResult.landmarks.qa.keypoint_confidences && (
                    <details className="bg-gray-900/80 rounded-xl border border-gray-700/50 overflow-hidden">
                      <summary className="px-4 py-3 text-sm font-medium text-gray-300 cursor-pointer hover:bg-gray-800/50 transition-colors">
                        キーポイント信頼度
                      </summary>
                      <div className="px-4 pb-4 space-y-2">
                        {(["humerus_shaft","lateral_epicondyle","medial_epicondyle","forearm_shaft","radial_head","olecranon"] as const).map((key, i) => {
                          const conf = analysisResult.landmarks.qa.keypoint_confidences![i];
                          if (conf === undefined) return null;
                          const pct = Math.round(conf * 100);
                          const color = conf > 0.7 ? "#22c55e" : conf > 0.4 ? "#eab308" : "#ef4444";
                          return (
                            <div key={key} className="flex items-center gap-2 text-xs">
                              <span className="w-16 text-gray-400 truncate" style={{ color: LANDMARK_COLORS[key] }}>
                                {LANDMARK_LABELS[key]}
                              </span>
                              <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
                              </div>
                              <span className="w-10 text-right font-mono text-gray-400">{pct}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </details>
                  )}
                </div>
              ) : !isAnalyzing ? (
                <div className="text-center text-gray-600 py-12">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4b5563" strokeWidth="1.5" strokeLinecap="round">
                      <path d="M12 2 L12 8 M12 8 L8 14 M12 8 L16 14 M8 14 L8 22 M16 14 L16 22" />
                    </svg>
                  </div>
                  <p className="text-base text-gray-500 font-medium">画像をアップロードして解析開始</p>
                  <p className="text-sm text-gray-600 mt-1">DICOM / PNG / JPEG 対応</p>
                  <div className="mt-6 text-left bg-gray-900/80 rounded-xl p-5 border border-gray-800/60 text-xs space-y-1.5">
                    <p className="text-gray-400 font-semibold mb-3 text-sm">測定項目</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-gray-500">
                      <p>-- 外反角 (Carrying Angle)</p>
                      <p>-- 屈曲角 (Flexion)</p>
                      <p>-- 回内外 (Pron./Supi.)</p>
                      <p>-- 内反外反 (Varus/Valgus)</p>
                    </div>
                    <div className="border-t border-gray-800 pt-2 mt-2 space-y-1 text-purple-500/80">
                      <p>-- ConvNeXt セカンドオピニオン</p>
                      <p>-- Grad-CAM XAI 可視化</p>
                      <p>-- エッジバリデーション</p>
                      <p>-- ポジショニング補正ガイダンス</p>
                    </div>
                  </div>
                </div>
              ) : null
            )}

            {/* === Grad-CAM タブ === */}
            {activeTab === "gradcam" && (
              <div className="space-y-4">
                <div className="flex gap-2 flex-wrap">
                  {GRADCAM_TARGETS.map(t => (
                    <button key={t.key}
                      onClick={() => { setGradCamTarget(t.key); setGradCamResult(null); }}
                      className={`py-1.5 px-3 rounded-lg text-xs font-medium transition-all ${
                        gradCamTarget === t.key
                          ? "bg-purple-700 text-white shadow-sm"
                          : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300"
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                  <button onClick={() => runGradCam(gradCamTarget)} disabled={isGradCam || !currentFile}
                    className="py-1.5 px-4 rounded-lg text-xs font-medium bg-purple-700 hover:bg-purple-600 text-white disabled:opacity-40 transition-all">
                    {isGradCam ? "生成中..." : "生成"}
                  </button>
                </div>

                {gradCamResult?.success ? (
                  <>
                    <div className="bg-gray-900/80 rounded-xl border border-gray-700/50 p-4">
                      <p className="text-xs text-gray-400 mb-3">{gradCamResult.note}</p>
                      <div className="grid grid-cols-2 gap-3">
                        {Object.entries(gradCamResult.predicted_angles).map(([k, v]) => (
                          <div key={k} className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/30">
                            <p className="text-xs text-gray-500 mb-1">{k}</p>
                            <p className="text-white font-mono font-bold text-lg">{v.toFixed(1)}°</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    {gradCamResult.raw_heatmap && (
                      <div>
                        <p className="text-xs text-gray-400 mb-2 font-medium">ヒートマップ (単体)</p>
                        <img src={gradCamResult.raw_heatmap} alt="GradCAM heatmap"
                          className="w-full rounded-xl border border-gray-700/50" />
                      </div>
                    )}
                  </>
                ) : gradCamResult && !gradCamResult.success ? (
                  <div className="bg-amber-950/30 border border-amber-700/50 rounded-xl p-5 text-center">
                    <p className="text-amber-300 font-medium text-sm mb-1">ConvNeXt モデル未訓練</p>
                    <p className="text-xs text-amber-400/70">elbow_convnext_best.pth を訓練・配置後に利用可能になります。</p>
                  </div>
                ) : !isGradCam ? (
                  <div className="text-center text-gray-500 py-10">
                    <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 8v4l3 3" />
                      </svg>
                    </div>
                    <p className="text-sm font-medium">「生成」ボタンで Grad-CAM XAI を実行</p>
                    <p className="text-xs mt-1 text-gray-600">ConvNeXt-Small が「どこを見て判断したか」を可視化</p>
                  </div>
                ) : (
                  <div className="text-center py-10">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-900/50 mb-3">
                      <svg className="animate-spin w-6 h-6 text-purple-400" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.3" />
                        <path d="M12 2 A10 10 0 0 1 22 12" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                      </svg>
                    </div>
                    <p className="text-purple-300 text-sm">Grad-CAM 生成中...</p>
                  </div>
                )}
              </div>
            )}

            {/* === 履歴タブ === */}
            {activeTab === "history" && (
              <HistoryPanel
                entries={history}
                onSelect={loadFromHistory}
                onClear={() => { setHistory([]); saveHistory([]); }}
              />
            )}

            {/* === 比較タブ === */}
            {activeTab === "compare" && (
              <CompareView
                entries={history}
                onClose={() => setActiveTab("analysis")}
              />
            )}
          </div>
        </div>
      </main>

      {/* フッター */}
      <footer className="border-t border-gray-800/50 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-600">
          <span>ElbowVision v1.0 -- AI Elbow Joint Angle Analysis System</span>
          <span>YOLOv8-Pose + ConvNeXt-Small + Classical CV + Grad-CAM XAI</span>
        </div>
      </footer>
    </div>
  );
}
