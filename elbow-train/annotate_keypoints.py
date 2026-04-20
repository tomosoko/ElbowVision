"""
実X線画像キーポイントアノテーションツール（6点対応）
使い方: python3 elbow-train/annotate_keypoints.py <画像パス> <出力JSONパス> [--model <model.pt>]

6点を順にクリック:
  1. humerus_shaft      (上腕骨骨幹)
  2. lateral_epicondyle (外側上顆)
  3. medial_epicondyle  (内側上顆)
  4. forearm_shaft      (前腕骨骨幹)
  5. radial_head        (橈骨頭)
  6. olecranon          (肘頭)

右クリックで1つ戻る、Enterで保存

--model オプション: 既存モデルで初期推定を表示（参考用）
"""
import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

KEYPOINT_NAMES = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "forearm_shaft",
    "radial_head",
    "olecranon",
]
N_KP = len(KEYPOINT_NAMES)
COLORS = ["#FF4444", "#44FF44", "#4488FF", "#FFAA00", "#00DDFF", "#FF00FF"]
INSTRUCTIONS = [
    "1/6: 上腕骨骨幹 (humerus_shaft) をクリック",
    "2/6: 外側上顆 (lateral_epicondyle) をクリック",
    "3/6: 内側上顆 (medial_epicondyle) をクリック",
    "4/6: 前腕骨骨幹 (forearm_shaft) をクリック",
    "5/6: 橈骨頭 (radial_head) をクリック",
    "6/6: 肘頭 (olecranon) をクリック",
    "完了！ Enterキーで保存",
]


def load_model_predictions(model_path, img_path, w, h):
    """Load model predictions as initial reference points."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.predict(source=img_path, imgsz=512, conf=0.1,
                                device="mps", verbose=False)
        r = results[0]
        if r.keypoints is not None and len(r.keypoints) > 0:
            kpts = r.keypoints.data[0].cpu().numpy()
            return [(float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2]))
                    for i in range(min(len(kpts), N_KP))]
    except Exception as e:
        print(f"Model prediction failed: {e}")
    return None


def load_existing_annotation(json_path):
    """Load existing annotation as starting points."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        kps = data.get("keypoints", {})
        w = data["width"]
        h = data["height"]
        result = []
        for name in KEYPOINT_NAMES:
            if name in kps:
                entry = kps[name]
                result.append((entry["x"], entry["y"]))
            else:
                result.append(None)
        return result, w, h
    except Exception as e:
        import sys
        print(f"[annotate_keypoints] アノテーション読み込み失敗: {e}", file=sys.stderr)
        return None, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Annotate 6 keypoints on X-ray")
    parser.add_argument("image", help="Path to X-ray image")
    parser.add_argument("output", help="Path to output JSON")
    parser.add_argument("--model", default=None,
                        help="Model path for initial predictions (reference)")
    args = parser.parse_args()

    img_path = args.image
    out_path = args.output

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    keypoints = []  # [(x, y), ...]
    markers = []
    texts = []

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(img, cmap="gray")
    ax.set_title(INSTRUCTIONS[0], fontsize=13, color="white",
                 backgroundcolor="black", pad=8)
    ax.axis("off")

    # Show model predictions as faint reference markers
    if args.model:
        preds = load_model_predictions(args.model, img_path, w, h)
        if preds:
            for i, (px, py, pc) in enumerate(preds):
                if pc > 0.1:
                    ax.plot(px, py, "x", color=COLORS[i], markersize=12,
                            markeredgewidth=1, alpha=0.4)
                    ax.text(px + w * 0.01, py + h * 0.01,
                            f"pred:{KEYPOINT_NAMES[i][:3]}",
                            color=COLORS[i], fontsize=7, alpha=0.4)

    # Also show existing annotation if available
    existing, _, _ = load_existing_annotation(out_path)
    if existing:
        for i, pt in enumerate(existing):
            if pt is not None:
                ax.plot(pt[0], pt[1], "s", color=COLORS[i], markersize=8,
                        markeredgecolor="white", markeredgewidth=1, alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS, KEYPOINT_NAMES)]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.7)

    def update_title():
        idx = len(keypoints)
        title = INSTRUCTIONS[min(idx, len(INSTRUCTIONS) - 1)]
        ax.set_title(title, fontsize=13, color="white",
                     backgroundcolor="black", pad=8)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click: add
            if len(keypoints) >= N_KP:
                return
            x, y = event.xdata, event.ydata
            keypoints.append((x, y))
            idx = len(keypoints) - 1
            m = ax.plot(x, y, "o", color=COLORS[idx], markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5)[0]
            t = ax.text(x + w * 0.01, y - h * 0.01, KEYPOINT_NAMES[idx],
                        color=COLORS[idx], fontsize=9,
                        backgroundcolor="black")
            markers.append(m)
            texts.append(t)
            update_title()
        elif event.button == 3:  # Right click: undo
            if keypoints:
                keypoints.pop()
                markers.pop().remove()
                texts.pop().remove()
                update_title()
                fig.canvas.draw()

    def on_key(event):
        if event.key == "enter" and len(keypoints) == N_KP:
            result = {
                "image": img_path,
                "width": w,
                "height": h,
                "keypoints": {
                    name: {"x": kp[0], "y": kp[1],
                           "x_norm": kp[0] / w, "y_norm": kp[1] / h}
                    for name, kp in zip(KEYPOINT_NAMES, keypoints)
                }
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"保存: {out_path}")
            for name, kp in zip(KEYPOINT_NAMES, keypoints):
                print(f"  {name}: ({kp[0]:.1f}, {kp[1]:.1f})")
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
