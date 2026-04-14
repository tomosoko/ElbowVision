"""
val画像を一括でAPI解析してCSVに出力するスクリプト

使い方:
  python elbow-train/batch_analyze.py \
    --input data/images/val/ \
    --output validation_output/ai_results.csv

前提: APIサーバーが http://localhost:8000 で起動済みであること
"""
import argparse
import csv
import os
import sys
import requests


def main():
    parser = argparse.ArgumentParser(description="ElbowVision 一括解析")
    parser.add_argument("--input",  required=True, help="解析対象画像ディレクトリ")
    parser.add_argument("--output", required=True, help="CSV出力先")
    parser.add_argument("--api",    default="http://localhost:8000", help="APIのベースURL")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ディレクトリが見つかりません: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    images = sorted([f for f in os.listdir(args.input)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"画像が見つかりません: {args.input}")
        sys.exit(1)

    print(f"{len(images)} 枚を解析します...")
    print(f"API: {args.api}")
    print()

    ok, skip = 0, 0

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "ai_carrying", "ai_flexion", "ai_pronation", "ai_varus_valgus",
            "ps_label", "vv_label",
            "view_type", "qa_score", "qa_status",
            "inference_engine",
        ])

        for fname in images:
            img_path = os.path.join(args.input, fname)
            image_id = os.path.splitext(fname)[0]

            try:
                with open(img_path, "rb") as img:
                    r = requests.post(
                        f"{args.api}/api/analyze",
                        files={"file": (fname, img, "image/png")},
                        timeout=30,
                    )
                if r.status_code != 200:
                    print(f"  SKIP: {fname}  (HTTP {r.status_code})")
                    skip += 1
                    continue

                d = r.json()
                a = d["landmarks"]["angles"]
                q = d["landmarks"]["qa"]

                writer.writerow([
                    image_id,
                    a["carrying_angle"],
                    a["flexion"],
                    a["pronation_sup"],
                    a["varus_valgus"],
                    a.get("ps_label", ""),
                    a.get("vv_label", ""),
                    q["view_type"],
                    q["score"],
                    q["status"],
                    q.get("inference_engine", "classical_cv"),
                ])
                carrying_str = f"{a['carrying_angle']:5.1f}" if a['carrying_angle'] is not None else "  N/A"
                flexion_str  = f"{a['flexion']:5.1f}" if a['flexion'] is not None else "  N/A"
                print(f"  OK: {fname:<30}  carrying={carrying_str}°  "
                      f"flexion={flexion_str}°  QA={q['score']}")
                ok += 1

            except requests.exceptions.ConnectionError:
                print(f"  ERROR: APIに接続できません。http://localhost:8000 が起動しているか確認してください。")
                sys.exit(1)
            except Exception as e:
                print(f"  SKIP: {fname}  ({e})")
                skip += 1

    print()
    print(f"完了: {ok} 枚成功 / {skip} 枚スキップ")
    print(f"出力: {args.output}")
    print()
    print("次のステップ:")
    print("  手動計測値を同じCSVに追記して bland_altman_analysis.py を実行")


if __name__ == "__main__":
    main()
