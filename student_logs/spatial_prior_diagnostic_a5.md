# Spatial-prior diagnostic (alpha=5.0)

Cases: run_1, run_50, run_100, run_200, run_300, run_400, run_485

## Pearson correlation (mean across cases)

| Quantity | Mean |
|---|---:|
| pearson(spatial_score, \|WSS\|) | +0.3149 |
| pearson(spatial_score, \|Cp\|)  | +0.4529 |

Green light (ρ ≥ +0.20): **PASS**

## Bin-occupancy (mean across cases)

| Bin | Uniform | Weighted | Oversample |
|---|---:|---:|---:|
| Top 10% | 10.00% | 14.80% | x1.48 |
| Bottom 50% | 50.00% | 36.39% | x0.73 |

## Per-case

| Case | N | rho(spatial,\|WSS\|) | rho(spatial,\|Cp\|) | top10%_mass | bot50%_mass |
|---|---:|---:|---:|---:|---:|
| run_1 | 8828095 | +0.3082 | +0.4350 | 15.24% | 35.57% |
| run_50 | 8194077 | +0.2846 | +0.4193 | 14.56% | 36.95% |
| run_100 | 8254434 | +0.3507 | +0.4730 | 14.55% | 36.49% |
| run_200 | 8377308 | +0.3282 | +0.4829 | 15.01% | 36.25% |
| run_300 | 8185956 | +0.3421 | +0.4757 | 14.48% | 36.67% |
| run_400 | 8701452 | +0.2994 | +0.4503 | 15.03% | 36.32% |
| run_485 | 9242495 | +0.2909 | +0.4343 | 14.75% | 36.45% |
