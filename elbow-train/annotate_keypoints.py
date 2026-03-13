"""
実X線画像キーポイントアノテーションツール
使い方: python3 elbow-train/annotate_keypoints.py <画像パス> <出力JSONパス>

4点を順にクリック:
  1. humerus_shaft      (上腕骨骨幹)
  2. lateral_epicondyle (外側上顆)
  3. medial_epicondyle  (内側上顆)
  4. forearm_shaft      (前腕骨骨幹)

右クリックで1つ戻る、Enterで保存
"""
import sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

KEYPOINT_NAMES = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "forearm_shaft",
]
COLORS = ["#FF4444", "#44FF44", "#4488FF", "#FFAA00"]
INSTRUCTIONS = [
    "1/4: 上腕骨骨幹 (humerus_shaft) をクリック",
    "2/4: 外側上顆 (lateral_epicondyle) をクリック",
    "3/4: 内側上顆 (medial_epicondyle) をクリック",
    "4/4: 前腕骨骨幹 (forearm_shaft) をクリック",
    "完了！ Enterキーで保存",
]

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 annotate_keypoints.py <image.png> <output.json>")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = sys.argv[2]

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

    # 凡例
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(COLORS, KEYPOINT_NAMES)]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0.7)

    def update_title():
        idx = len(keypoints)
        title = INSTRUCTIONS[min(idx, len(INSTRUCTIONS)-1)]
        ax.set_title(title, fontsize=13, color="white",
                     backgroundcolor="black", pad=8)
        fig.canvas.draw()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # 左クリック: 追加
            if len(keypoints) >= 4:
                return
            x, y = event.xdata, event.ydata
            keypoints.append((x, y))
            idx = len(keypoints) - 1
            m = ax.plot(x, y, "o", color=COLORS[idx], markersize=10,
                        markeredgecolor="white", markeredgewidth=1.5)[0]
            t = ax.text(x + w*0.01, y - h*0.01, KEYPOINT_NAMES[idx],
                        color=COLORS[idx], fontsize=9,
                        backgroundcolor="black")
            markers.append(m)
            texts.append(t)
            update_title()
        elif event.button == 3:  # 右クリック: 戻る
            if keypoints:
                keypoints.pop()
                markers.pop().remove()
                texts.pop().remove()
                update_title()
                fig.canvas.draw()

    def on_key(event):
        if event.key == "enter" and len(keypoints) == 4:
            result = {
                "image": img_path,
                "width": w,
                "height": h,
                "keypoints": {
                    name: {"x": kp[0], "y": kp[1],
                           "x_norm": kp[0]/w, "y_norm": kp[1]/h}
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
