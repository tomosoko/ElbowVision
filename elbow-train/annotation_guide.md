# LabelStudio アノテーションガイド — ElbowVision

## キーポイント定義（4点）

| ID | 名前 | 位置 |
|----|------|------|
| 0 | humerus_shaft | 上腕骨幹部（近位端: 骨幹の上端中央） |
| 1 | lateral_epicondyle | 外側上顆（関節外側の突出部） |
| 2 | medial_epicondyle | 内側上顆（関節内側の突出部） |
| 3 | forearm_shaft | 前腕骨幹部（遠位端: 尺骨/橈骨の下端中央） |

## AP像（正面）でのアノテーション

- **humerus_shaft**: 上腕骨の上端（骨幹中央線上）
- **lateral_epicondyle**: 外側上顆の最突出点
- **medial_epicondyle**: 内側上顆の最突出点
- **forearm_shaft**: 尺骨幹部の遠位端（中央）

## 側面像でのアノテーション

- **humerus_shaft**: 上腕骨後方皮質の上端
- **lateral_epicondyle**: 外側上顆の最外側点（AP像と同じ骨の点。側面像では上顆が重なって見えるが、外側の膨らみを選ぶ）
- **medial_epicondyle**: 内側上顆の最後内側点（AP像と同じ骨の点。側面像では後方に張り出した内側上顆の先端）
- **forearm_shaft**: 橈骨骨幹の遠位端

> **注意**: 側面像では外側上顆と内側上顆は重なって見えるが、小頭（capitulum）や肘頭（olecranon）を代替として使わないこと。
> AP像・側面像で同じ解剖学的点を一貫してマーキングすることで、モデルが両ビューで整合したキーポイントを学習できる。

## YOLO ラベル形式

```
<class_id> <bbox_cx> <bbox_cy> <bbox_w> <bbox_h> <kp0_x> <kp0_y> <kp0_vis> <kp1_x> <kp1_y> <kp1_vis> <kp2_x> <kp2_y> <kp2_vis> <kp3_x> <kp3_y> <kp3_vis>
```

- 全座標は画像サイズで正規化（0〜1）
- vis: 0=不可視, 1=隠れている, 2=可視

## LabelStudio エクスポート設定

1. プロジェクト設定 → Export → YOLO format
2. キーポイント数: 4
3. `flip_idx: [0, 2, 1, 3]`（外側・内側上顆を左右入れ替え）

## 推奨アノテーション数

- 最低: AP像 30枚 + 側面像 30枚
- 推奨: AP像 100枚 + 側面像 100枚
- train/val = 80/20 分割
