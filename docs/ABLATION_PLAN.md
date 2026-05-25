# Ablation Plan

| Experiment | Variable | Fixed Controls | Metric |
|---|---|---|---|
| baseline | schema checks only | dataset | flagged precision |
| spatial thresholds | limb ratio/margin | dataset | precision/recall |
| temporal thresholds | max velocity/window | ordered frames | false positive rate |
| combined QA | schema + spatial + temporal | reviewer labels | workload reduction |
| visualization review | with/without overlays | reviewer set | review time |

Each ablation should include config, reviewer label source, metrics CSV, and example flagged cases.
