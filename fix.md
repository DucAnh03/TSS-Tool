|
45 | f"| FAR | {fmt(v.get('far'), 4)} |",
46 | f"| FRR | {fmt(v.get('frr'), 4)} |",
47 | f"",
| ^^^
48 | ]
|
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> scripts/gen_report.py:56:9
|
55 | lines += [
56 | f"## Vision Pipeline (CIFAR-10)",
| ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
57 | f"",
58 | f"| Metric | Value |",
|
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> scripts/gen_report.py:57:9
|
55 | lines += [
56 | f"## Vision Pipeline (CIFAR-10)",
57 | f"",
| ^^^
58 | f"| Metric | Value |",
59 | f"|--------|-------|",
|
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> scripts/gen_report.py:58:9
|
56 | f"## Vision Pipeline (CIFAR-10)",
57 | f"",
58 | f"| Metric | Value |",
| ^^^^^^^^^^^^^^^^^^^^^
59 | f"|--------|-------|",
60 | f"| dataset | {fmt(vis.get('dataset'))} |",
|
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> scripts/gen_report.py:59:9
|
57 | f"",
58 | f"| Metric | Value |",
59 | f"|--------|-------|",
| ^^^^^^^^^^^^^^^^^^^^^
60 | f"| dataset | {fmt(vis.get('dataset'))} |",
61 | f"| model | {fmt(vis.get('model'))} |",
|
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> scripts/gen_report.py:64:9
|
62 | f"| accuracy | {fmt(vis.get('accuracy'), 4)} |",
63 | f"| loss | {fmt(vis.get('loss'), 4)} |",
64 | f"",
| ^^^
65 | ]
|
help: Remove extraneous `f` prefix

F401 [*] `re` imported but unused
--> scripts/kaggle_trigger.py:9:8
|
7 | import json
8 | import os
9 | import re
| ^^
10 | import shutil
11 | import subprocess
|
help: Remove unused import: `re`

F541 [*] f-string without any placeholders
--> train.py:64:13
|
62 | f.write(f"- **Train samples:** {TRAIN_SIZE}\n")
63 | f.write(f"- **Test samples:** {TEST_SIZE}\n")
64 | f.write(f"- **Dataset:** dair-ai/emotion (HuggingFace)\n")
| ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
65 | f.write(f"- **Model:** TF-IDF (5k features, bigrams) + LogisticRegression\n\n")
66 | f.write("### Per-class Report\n\n```\n")
   |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
--> train.py:65:13
|
63 | f.write(f"- **Test samples:** {TEST_SIZE}\n")
64 | f.write(f"- **Dataset:** dair-ai/emotion (HuggingFace)\n")
65 | f.write(f"- **Model:** TF-IDF (5k features, bigrams) + LogisticRegression\n\n")
| ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
66 | f.write("### Per-class Report\n\n```\n")
67 |     f.write(report)
   |
help: Remove extraneous `f` prefix

E402 Module level import not at top of file
--> train.py:82:1
|
81 | # ── 6. Update metrics.json contract ──────────────────────────────────────────
82 | import subprocess
| ^^^^^^^^^^^^^^^^^
83 | from datetime import datetime, timezone
|

E402 Module level import not at top of file
--> train.py:83:1
|
81 | # ── 6. Update metrics.json contract ──────────────────────────────────────────
82 | import subprocess
83 | from datetime import datetime, timezone
| ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
84 |
85 | try:
|

F401 [*] `numpy` imported but unused
--> vision/train.py:110:21
|
108 | from sklearn.metrics import confusion_matrix
109 | import matplotlib.pyplot as plt
110 | import numpy as np
| ^^
111 |
112 | cm = confusion_matrix(labels, preds)
|
help: Remove unused import: `numpy`

E702 Multiple statements on one line (semicolon)
--> vision/train.py:115:29
|
113 | fig, ax = plt.subplots(figsize=(8, 7))
114 | im = ax.imshow(cm, cmap="Blues")
115 | ax.set_xticks(range(10)); ax.set_yticks(range(10))
| ^
116 | ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
117 | ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
|

E702 Multiple statements on one line (semicolon)
--> vision/train.py:121:31
|
119 | for j in range(10):
120 | ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
121 | ax.set_xlabel("Predicted"); ax.set_ylabel("True")
| ^
122 | ax.set_title("CIFAR-10 Confusion Matrix")
123 | plt.colorbar(im, ax=ax)
|

E702 Multiple statements on one line (semicolon)
--> voice/client.py:234:48
|
232 | def main():
233 | if not VPS_IP:
234 | print("[!] VPS_IP chưa đặt trong .env"); sys.exit(1)
| ^
235 | if not API_KEY:
236 | print("[!] CLIENT_API_KEY chưa đặt trong .env"); sys.exit(1)
|

E702 Multiple statements on one line (semicolon)
--> voice/client.py:236:56
|
234 | print("[!] VPS_IP chưa đặt trong .env"); sys.exit(1)
235 | if not API_KEY:
236 | print("[!] CLIENT_API_KEY chưa đặt trong .env"); sys.exit(1)
| ^
237 |
238 | # Start debug WS server in background thread
|

F541 [*] f-string without any placeholders
--> voice/client.py:258:11
|
257 | print(f"\n\033[96m[Sẵn sàng]\033[0m Nghe: '\033[1m{WAKE_MODEL}\033[0m'")
258 | print(f"[Dashboard] http://localhost:3000\n")
| ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
259 |
260 | frame_n = 0
|
help: Remove extraneous `f` prefix

F401 [*] `os` imported but unused
--> voice/eval.py:10:8
|
8 | import argparse
9 | import json
10 | import os
| ^^
11 | from pathlib import Path
|
help: Remove unused import: `os`

Found 50 errors.
[*] 29 fixable with the `--fix` option.
Error: Process completed with exit code 1.
