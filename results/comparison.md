# Comparison (eval_test_sipakmed-herlev)

### Test subset: all (n = 748)

| Metric | densenet121 (3 seeds) | resnet50 (3 seeds) |
| --- | --- | --- |
| Accuracy (%) | 94.70 ± 0.67 | 95.01 ± 0.20 |
| Macro F1 | 0.9467 ± 0.0068 | 0.9500 ± 0.0021 |
| ROC AUC | 0.9910 ± 0.0024 | 0.9895 ± 0.0048 |
| Recall abnormal (%) | 94.83 ± 2.01 | 96.84 ± 1.49 |
| Recall normal (%) | 94.58 ± 0.88 | 93.42 ± 1.04 |
| Precision abnormal (%) | 93.85 ± 0.85 | 92.77 ± 0.95 |
| Best epoch | 14.0 ± 5.3 | 14.7 ± 3.5 |
| Training time (min) | 13.61 ± 4.19 | 8.07 ± 1.52 |
| Inference (ms / image) | 12.33 ± 0.38 | 5.69 ± 0.07 |
| Parameters | 7.0 M | 23.5 M |

### Test subset: sipakmed (n = 609)

| Metric | densenet121 (3 seeds) | resnet50 (3 seeds) |
| --- | --- | --- |
| Accuracy (%) | 96.28 ± 0.58 | 96.72 ± 0.28 |
| Macro F1 | 0.9614 ± 0.0059 | 0.9660 ± 0.0029 |
| ROC AUC | 0.9954 ± 0.0009 | 0.9950 ± 0.0032 |
| Recall abnormal (%) | 96.05 ± 1.03 | 97.55 ± 1.08 |
| Recall normal (%) | 96.43 ± 0.99 | 96.15 ± 0.73 |
| Precision abnormal (%) | 94.78 ± 1.36 | 94.48 ± 0.94 |

### Test subset: herlev (n = 139)

| Metric | densenet121 (3 seeds) | resnet50 (3 seeds) |
| --- | --- | --- |
| Accuracy (%) | 87.77 ± 3.14 | 87.53 ± 1.50 |
| Macro F1 | 0.8411 ± 0.0348 | 0.8254 ± 0.0115 |
| ROC AUC | 0.9355 ± 0.0161 | 0.9178 ± 0.0283 |
| Recall abnormal (%) | 91.91 ± 4.48 | 95.15 ± 3.88 |
| Recall normal (%) | 75.93 ± 1.60 | 65.74 ± 5.78 |
| Precision abnormal (%) | 91.61 ± 0.47 | 88.87 ± 1.33 |
