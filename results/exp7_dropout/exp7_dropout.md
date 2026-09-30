# Exp 7 — Whole-sensor failure at test time (Car mAP, test scenes)


## clear0

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.800 | 0.780 | 0.186 | 0.795 |
| Fixed weights | 0.807 | 0.783 | 0.216 | 0.801 |
| Per-frame weather classifier | 0.804 | 0.783 | 0.196 | 0.798 |
| Per-object (ours) | 0.865 | 0.851 | 0.293 | 0.798 |
## fog3

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.303 | 0.292 | 0.029 | 0.292 |
| Fixed weights | 0.310 | 0.295 | 0.031 | 0.297 |
| Per-frame weather classifier | 0.314 | 0.297 | 0.030 | 0.298 |
| Per-object (ours) | 0.404 | 0.387 | 0.089 | 0.301 |
## snow3

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.698 | 0.669 | 0.154 | 0.690 |
| Fixed weights | 0.709 | 0.674 | 0.178 | 0.697 |
| Per-frame weather classifier | 0.703 | 0.673 | 0.168 | 0.692 |
| Per-object (ours) | 0.785 | 0.763 | 0.242 | 0.699 |
## night2

| arm | all sensors | camera failed | LiDAR failed | radar failed |
|---|---|---|---|---|
| Fixed (clear-fit, status quo) | 0.797 | 0.780 | 0.121 | 0.792 |
| Fixed weights | 0.805 | 0.783 | 0.145 | 0.799 |
| Per-frame weather classifier | 0.802 | 0.780 | 0.169 | 0.797 |
| Per-object (ours) | 0.855 | 0.858 | 0.215 | 0.797 |