# Spatial-prior diagnostic (alpha=3.0)

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
| Top 10% | 10.00% | 13.91% | x1.39 |
| Bottom 50% | 50.00% | 38.91% | x0.78 |

## Per-case

| Case | N | rho(spatial,\|WSS\|) | rho(spatial,\|Cp\|) | top10%_mass | bot50%_mass |
|---|---:|---:|---:|---:|---:|
| run_1 | 8828095 | +0.3082 | +0.4350 | 14.26% | 38.28% |
| run_50 | 8194077 | +0.2846 | +0.4193 | 13.72% | 39.37% |
| run_100 | 8254434 | +0.3507 | +0.4730 | 13.71% | 38.99% |
| run_200 | 8377308 | +0.3282 | +0.4829 | 14.07% | 38.84% |
| run_300 | 8185956 | +0.3421 | +0.4757 | 13.66% | 39.11% |
| run_400 | 8701452 | +0.2994 | +0.4503 | 14.09% | 38.87% |
| run_485 | 9242495 | +0.2909 | +0.4343 | 13.88% | 38.95% |
