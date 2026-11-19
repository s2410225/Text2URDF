# Cross-morphology results (mean ± s.e. over seeds, greedy eval at the end)

## keypoint space

| mode | n | radial eval | radial dx [m] | bilateral eval | bilateral dx [m] | radial teaches | bilateral policy on radial body dx |
|---|---|---|---|---|---|---|---|
| indep | 4 | 3711.3 ± 71.2 | 152.1 ± 3.4 | 1743.0 ± 5.3 | 50.0 ± 0.2 | 0% | -0.9 ± 0.2 |
| mutual | 4 | 1458.9 ± 120.1 | 23.9 ± 6.2 | 987.5 ± 2.5 | -0.0 ± 0.0 | 0% | 0.2 ± 0.0 |
| better | 4 | 3711.3 ± 71.2 | 152.1 ± 3.4 | 802.8 ± 15.7 | 0.1 ± 0.0 | 98% | 54.0 ± 11.9 |
| better_q | 4 | 3711.3 ± 71.2 | 152.1 ± 3.4 | 805.0 ± 17.5 | 0.2 ± 0.1 | 98% | 12.7 ± 8.1 |

## joint space

| mode | n | radial eval | radial dx [m] | bilateral eval | bilateral dx [m] | radial teaches | bilateral policy on radial body dx |
|---|---|---|---|---|---|---|---|
| indep | 2 | 2892.4 ± 567.0 | 121.1 ± 33.1 | 2737.7 ± 98.0 | 118.0 ± 11.7 | 0% | 0.4 ± 0.4 |
| better | 2 | 2834.6 ± 905.6 | 116.8 ± 42.4 | 853.3 ± 207.2 | 6.8 ± 2.6 | 64% | 26.9 ± 26.8 |
