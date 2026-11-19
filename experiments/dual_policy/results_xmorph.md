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
| indep | 3 | 3101.1 ± 388.2 | 131.7 ± 21.8 | 2326.6 ± 415.0 | 95.3 ± 23.6 | 0% | 0.7 ± 0.4 |
| better | 3 | 2916.2 ± 529.2 | 117.7 ± 24.5 | 1004.4 ± 192.8 | 12.5 ± 5.9 | 66% | 19.3 ± 17.3 |
