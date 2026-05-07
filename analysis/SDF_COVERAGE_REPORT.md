# SDF Train-set Coverage Diagnostic

- Splits: train=400, val=34, test=50
- OOD cases under inspection: ['run_133', 'run_158', 'run_203', 'run_226']
- k-NN k = 5
- Histogram bins: 32, range = (-2.0, 5.0)
- Mahalanobis covariance: estimated from train scalars over ('sdf_mean', 'sdf_std', 'sdf_min', 'sdf_max')

## OOD-4 coverage summary

| case_id | knn_4d_mahal | z_vs_other_test | knn_hist_chi2 | z_vs_other_test | extrapolative_2σ |
| ------- | ------------ | --------------- | ------------- | --------------- | ---------------- |
| run_133 | 1.166        | +1.05           | 0.000145      | +3.66           | YES              |
| run_158 | 0.734        | +0.41           | 0.0001184     | +2.49           | YES              |
| run_203 | 0.441        | -0.03           | 0.0001223     | +2.67           | YES              |
| run_226 | 0.880        | +0.62           | 0.0001539     | +4.05           | YES              |

- Other 46 test (4D-Mahal): mean=0.464, std=0.666
- Other 46 test (hist-χ²): mean=6.154e-05, std=2.282e-05

## All test cases ranked by 4D-Mahal k-NN distance to train

| rank | case_id | knn_4d_mahal | knn_hist_chi2 | is_OOD4 |
| ---- | ------- | ------------ | ------------- | ------- |
| 1    | run_387 | 4.843        | 0.0001218     |         |
| 2    | run_133 | 1.166        | 0.000145      | *       |
| 3    | run_226 | 0.880        | 0.0001539     | *       |
| 4    | run_108 | 0.850        | 7.39e-05      |         |
| 5    | run_19  | 0.774        | 4.448e-05     |         |
| 6    | run_158 | 0.734        | 0.0001184     | *       |
| 7    | run_12  | 0.600        | 3.415e-05     |         |
| 8    | run_258 | 0.573        | 7.709e-05     |         |
| 9    | run_472 | 0.552        | 4.553e-05     |         |
| 10   | run_215 | 0.467        | 7.691e-05     |         |
| 11   | run_173 | 0.444        | 0.0001064     |         |
| 12   | run_26  | 0.441        | 4.628e-05     |         |
| 13   | run_203 | 0.441        | 0.0001223     | *       |
| 14   | run_41  | 0.433        | 5.413e-05     |         |
| 15   | run_405 | 0.426        | 5.405e-05     |         |
| 16   | run_222 | 0.420        | 5.107e-05     |         |
| 17   | run_11  | 0.414        | 8.205e-05     |         |
| 18   | run_428 | 0.392        | 3.754e-05     |         |
| 19   | run_208 | 0.391        | 6.547e-05     |         |
| 20   | run_280 | 0.387        | 6.36e-05      |         |
| 21   | run_284 | 0.383        | 3.735e-05     |         |
| 22   | run_410 | 0.382        | 4.31e-05      |         |
| 23   | run_127 | 0.361        | 6.087e-05     |         |
| 24   | run_436 | 0.357        | 6.562e-05     |         |
| 25   | run_24  | 0.355        | 9.802e-05     |         |
| 26   | run_197 | 0.351        | 7.947e-05     |         |
| 27   | run_55  | 0.350        | 5.347e-05     |         |
| 28   | run_29  | 0.345        | 0.0001335     |         |
| 29   | run_180 | 0.334        | 5.779e-05     |         |
| 30   | run_322 | 0.332        | 5.185e-05     |         |
| 31   | run_210 | 0.321        | 5.102e-05     |         |
| 32   | run_350 | 0.321        | 4.395e-05     |         |
| 33   | run_207 | 0.314        | 5.043e-05     |         |
| 34   | run_205 | 0.303        | 6.093e-05     |         |
| 35   | run_424 | 0.302        | 7.753e-05     |         |
| 36   | run_59  | 0.300        | 4.382e-05     |         |
| 37   | run_354 | 0.286        | 5.04e-05      |         |
| 38   | run_382 | 0.281        | 5.078e-05     |         |
| 39   | run_142 | 0.280        | 9.535e-05     |         |
| 40   | run_263 | 0.278        | 8.625e-05     |         |
| 41   | run_187 | 0.277        | 6.512e-05     |         |
| 42   | run_372 | 0.269        | 7.084e-05     |         |
| 43   | run_124 | 0.267        | 3.769e-05     |         |
| 44   | run_290 | 0.264        | 4.101e-05     |         |
| 45   | run_337 | 0.263        | 4.12e-05      |         |
| 46   | run_20  | 0.258        | 7.535e-05     |         |
| 47   | run_429 | 0.233        | 4.224e-05     |         |
| 48   | run_363 | 0.189        | 5.312e-05     |         |
| 49   | run_188 | 0.188        | 2.028e-05     |         |
| 50   | run_199 | 0.172        | 5.802e-05     |         |

## Top 10 train cases by self-k-NN-distance (intra-train density check)

| rank | train_case_id | knn_4d_mahal | knn_hist_chi2 |
| ---- | ------------- | ------------ | ------------- |
| 1    | run_391       | 19.976       | 4.833e-05     |
| 2    | run_72        | 6.448        | 4.784e-05     |
| 3    | run_394       | 3.775        | 4.181e-05     |
| 4    | run_76        | 2.402        | 9.887e-05     |
| 5    | run_129       | 1.528        | 0.000168      |
| 6    | run_310       | 1.202        | 0.0001717     |
| 7    | run_498       | 1.154        | 4.07e-05      |
| 8    | run_404       | 1.139        | 4.797e-05     |
| 9    | run_183       | 1.120        | 6.573e-05     |
| 10   | run_373       | 1.064        | 0.0002356     |

## Per-case scalar stats (top 5 train, all 50 test)

| split | case_id | sdf_mean | sdf_std | sdf_min    | sdf_max | sdf_q05   | sdf_q25  | sdf_q50  | sdf_q75 | sdf_q95 | sdf_negative_frac | sdf_near_surface_frac |
| ----- | ------- | -------- | ------- | ---------- | ------- | --------- | -------- | -------- | ------- | ------- | ----------------- | --------------------- |
| train | run_1   | 0.2179   | 1.578   | -0.3162    | 80.26   | 0.0003631 | 0.001622 | 0.004895 | 0.03505 | 0.8978  | 0.0001324         | 0.8317                |
| train | run_2   | 0.2242   | 1.646   | -0.3196    | 530.4   | 0.0003675 | 0.001653 | 0.005119 | 0.03464 | 0.8688  | 0.0001412         | 0.8381                |
| train | run_3   | 0.2219   | 1.61    | -0.3804    | 80.5    | 0.0003587 | 0.00157  | 0.004645 | 0.02645 | 0.8848  | 0.0001846         | 0.8464                |
| train | run_5   | 0.2382   | 1.677   | -0.3516    | 80.42   | 0.0003675 | 0.001662 | 0.005378 | 0.0368  | 0.9956  | 0.00014           | 0.8311                |
| train | run_6   | 0.1923   | 1.52    | -0.2868    | 80.45   | 0.0003527 | 0.001536 | 0.004046 | 0.02043 | 0.6825  | 0.0001756         | 0.8587                |
| test  | run_133 | 0.1871   | 1.495   | -0.01483   | 80.1    | 0.0003547 | 0.001546 | 0.004143 | 0.02242 | 0.6541  | 1.253e-05         | 0.8543                |
| test  | run_158 | 0.19     | 1.496   | -0.001321  | 80.69   | 0.0003525 | 0.001537 | 0.004061 | 0.02006 | 0.6977  | 1.661e-05         | 0.8601                |
| test  | run_203 | 0.2011   | 1.515   | -0.001323  | 80.46   | 0.0003537 | 0.001547 | 0.00408  | 0.02135 | 0.8005  | 1.162e-05         | 0.8534                |
| test  | run_226 | 0.2048   | 1.517   | -0.0009287 | 80.6    | 0.000354  | 0.001554 | 0.004162 | 0.02333 | 0.8411  | 1.601e-05         | 0.8485                |
| test  | run_108 | 0.2087   | 1.537   | -0.3026    | 196.8   | 0.0003574 | 0.001571 | 0.004422 | 0.02592 | 0.8747  | 0.000157          | 0.8451                |
| test  | run_11  | 0.2409   | 1.661   | -0.4453    | 137.2   | 0.0003722 | 0.001705 | 0.005807 | 0.04517 | 0.9863  | 0.000116          | 0.8196                |
| test  | run_12  | 0.2105   | 1.565   | -0.4206    | 80.57   | 0.0003546 | 0.001556 | 0.00431  | 0.02304 | 0.8662  | 0.000171          | 0.8516                |
| test  | run_124 | 0.2156   | 1.597   | -0.3664    | 80.17   | 0.000362  | 0.001603 | 0.004859 | 0.02824 | 0.8708  | 0.0001792         | 0.8469                |
| test  | run_127 | 0.2416   | 1.656   | -0.3059    | 80.61   | 0.0003681 | 0.001663 | 0.005221 | 0.04159 | 1.026   | 0.0001428         | 0.8223                |
| test  | run_142 | 0.2212   | 1.623   | -0.2936    | 80.34   | 0.0003538 | 0.001543 | 0.004498 | 0.02395 | 0.8999  | 0.0001783         | 0.8513                |
| test  | run_173 | 0.2484   | 1.662   | -0.3143    | 80.58   | 0.0003686 | 0.00168  | 0.00539  | 0.03993 | 1.09    | 0.0001455         | 0.8228                |
| test  | run_180 | 0.232    | 1.613   | -0.3186    | 80.75   | 0.0003697 | 0.00169  | 0.005359 | 0.04011 | 1.003   | 0.0001449         | 0.8257                |
| test  | run_187 | 0.2212   | 1.598   | -0.3313    | 79.87   | 0.0003665 | 0.001649 | 0.005043 | 0.03461 | 0.8926  | 0.0001419         | 0.8338                |
| test  | run_188 | 0.2314   | 1.636   | -0.3191    | 80.55   | 0.0003608 | 0.001598 | 0.004904 | 0.03031 | 0.9736  | 0.0001662         | 0.8392                |
| test  | run_19  | 0.234    | 1.613   | -0.4289    | 86.61   | 0.0003642 | 0.001631 | 0.004983 | 0.0346  | 1.023   | 0.0001471         | 0.8314                |
| test  | run_197 | 0.2016   | 1.536   | -0.2902    | 80.16   | 0.0003581 | 0.00158  | 0.00451  | 0.02657 | 0.7876  | 0.0001554         | 0.8457                |
| test  | run_199 | 0.2324   | 1.649   | -0.3072    | 80.85   | 0.0003665 | 0.001641 | 0.005087 | 0.03437 | 0.9536  | 0.0001584         | 0.8347                |
| test  | run_20  | 0.2261   | 1.617   | -0.2852    | 100.2   | 0.0003639 | 0.001628 | 0.005048 | 0.03799 | 0.8989  | 0.0001356         | 0.8274                |
| test  | run_205 | 0.2376   | 1.641   | -0.3306    | 80.49   | 0.0003692 | 0.00168  | 0.005288 | 0.03912 | 1       | 0.0001413         | 0.8255                |
| test  | run_207 | 0.2297   | 1.648   | -0.275     | 80.58   | 0.000361  | 0.0016   | 0.004925 | 0.03282 | 0.8935  | 0.0001468         | 0.836                 |
| test  | run_208 | 0.2068   | 1.542   | -0.3715    | 80.46   | 0.0003521 | 0.001542 | 0.004272 | 0.02222 | 0.8556  | 0.0001872         | 0.8542                |
| test  | run_210 | 0.2157   | 1.584   | -0.2786    | 80.85   | 0.0003582 | 0.001576 | 0.004581 | 0.02686 | 0.8728  | 0.000164          | 0.8449                |
| test  | run_215 | 0.2494   | 1.679   | -0.3052    | 80.69   | 0.000371  | 0.00169  | 0.005683 | 0.04234 | 1.077   | 0.0001296         | 0.8214                |
| test  | run_222 | 0.2033   | 1.554   | -0.3582    | 80.46   | 0.000351  | 0.001524 | 0.004115 | 0.02056 | 0.7859  | 0.000209          | 0.8587                |
| test  | run_24  | 0.2309   | 1.642   | -0.3527    | 80.21   | 0.0003706 | 0.001687 | 0.005665 | 0.04002 | 0.9417  | 0.0001389         | 0.8265                |
| test  | run_258 | 0.224    | 1.58    | -0.4272    | 80.73   | 0.0003608 | 0.001602 | 0.004728 | 0.03248 | 0.9653  | 0.000141          | 0.8336                |
| test  | run_26  | 0.2233   | 1.626   | -0.4129    | 80.6    | 0.0003598 | 0.001593 | 0.004765 | 0.02897 | 0.902   | 0.0001559         | 0.8407                |
| test  | run_263 | 0.2429   | 1.698   | -0.3118    | 224.4   | 0.000369  | 0.001679 | 0.005586 | 0.04251 | 0.9464  | 0.0001251         | 0.8224                |
| test  | run_280 | 0.2053   | 1.551   | -0.3717    | 79.4    | 0.0003651 | 0.001632 | 0.004917 | 0.03134 | 0.8001  | 0.0001489         | 0.8393                |
| test  | run_284 | 0.2121   | 1.587   | -0.3311    | 80.67   | 0.0003583 | 0.001573 | 0.004653 | 0.026   | 0.8334  | 0.0001751         | 0.8474                |
| test  | run_29  | 0.2137   | 1.555   | -0.3221    | 80.75   | 0.0003528 | 0.001545 | 0.004188 | 0.0241  | 0.8731  | 0.0001455         | 0.8466                |
| test  | run_290 | 0.2292   | 1.624   | -0.2879    | 80.88   | 0.000363  | 0.001616 | 0.004935 | 0.03392 | 0.9623  | 0.0001491         | 0.8344                |
| test  | run_322 | 0.2163   | 1.598   | -0.2738    | 80.51   | 0.00036   | 0.001584 | 0.004728 | 0.03007 | 0.8394  | 0.0001515         | 0.8412                |
| test  | run_337 | 0.2141   | 1.587   | -0.3442    | 80.12   | 0.0003622 | 0.001613 | 0.004842 | 0.03158 | 0.8472  | 0.000161          | 0.8371                |
| test  | run_350 | 0.2221   | 1.595   | -0.3213    | 123.5   | 0.0003654 | 0.001636 | 0.005014 | 0.03552 | 0.9153  | 0.0001517         | 0.8323                |
| test  | run_354 | 0.2139   | 1.602   | -0.3188    | 80.54   | 0.0003619 | 0.001601 | 0.004843 | 0.02866 | 0.8184  | 0.0001567         | 0.8458                |
| test  | run_363 | 0.2295   | 1.631   | -0.2863    | 80.33   | 0.0003688 | 0.001666 | 0.005158 | 0.03822 | 0.9186  | 0.0001341         | 0.8298                |
| test  | run_372 | 0.2257   | 1.632   | -0.2733    | 80.04   | 0.0003632 | 0.001615 | 0.005019 | 0.03627 | 0.8882  | 0.0001334         | 0.831                 |
| test  | run_382 | 0.2233   | 1.598   | -0.3297    | 80.39   | 0.0003701 | 0.001691 | 0.005325 | 0.03803 | 0.9234  | 0.0001426         | 0.8292                |
| test  | run_387 | 0.2519   | 1.672   | -0.2949    | 676     | 0.0003705 | 0.001696 | 0.005622 | 0.04535 | 1.113   | 0.0001369         | 0.8155                |
| test  | run_405 | 0.217    | 1.577   | -0.3814    | 80.72   | 0.000355  | 0.001567 | 0.004404 | 0.02577 | 0.8718  | 0.0001572         | 0.8455                |
| test  | run_41  | 0.2087   | 1.578   | -0.3277    | 80.59   | 0.0003624 | 0.001612 | 0.004893 | 0.02873 | 0.8329  | 0.0001677         | 0.8462                |
| test  | run_410 | 0.2095   | 1.581   | -0.3185    | 80.72   | 0.000362  | 0.001595 | 0.004737 | 0.02932 | 0.7829  | 0.0001535         | 0.8436                |
| test  | run_424 | 0.2181   | 1.606   | -0.3634    | 79.95   | 0.0003649 | 0.001623 | 0.00503  | 0.03548 | 0.8246  | 0.0001359         | 0.8329                |
| test  | run_428 | 0.2115   | 1.598   | -0.3498    | 80.33   | 0.0003621 | 0.0016   | 0.004849 | 0.02882 | 0.8094  | 0.0001671         | 0.8449                |
| test  | run_429 | 0.226    | 1.612   | -0.3048    | 80.7    | 0.0003605 | 0.001592 | 0.004735 | 0.03032 | 0.937   | 0.0001479         | 0.8387                |
| test  | run_436 | 0.2175   | 1.577   | -0.4295    | 79.82   | 0.0003619 | 0.001603 | 0.004831 | 0.03387 | 0.8861  | 0.0001395         | 0.8336                |
| test  | run_472 | 0.2184   | 1.637   | -0.2993    | 80.12   | 0.0003608 | 0.001587 | 0.00485  | 0.02917 | 0.8304  | 0.0001547         | 0.8436                |
| test  | run_55  | 0.2053   | 1.552   | -0.343     | 80.06   | 0.0003516 | 0.001536 | 0.004062 | 0.02072 | 0.813   | 0.0001739         | 0.8558                |
| test  | run_59  | 0.2356   | 1.654   | -0.3248    | 80.89   | 0.0003641 | 0.001626 | 0.005021 | 0.03503 | 0.9772  | 0.0001364         | 0.8322                |

## Restored-case clustering analysis

Cases with `sdf_min > -0.05` (anomalously near-zero, suggesting a different SDF sampling pipeline that omitted inside-body samples):

- Train (6/400 = 1.5%): ['run_184', 'run_249', 'run_310', 'run_416', 'run_44', 'run_484']
- Test (4/50 = 8.0%): ['run_133', 'run_158', 'run_203', 'run_226']

Per-OOD-test case: 5 nearest train cases in 4D-Mahalanobis space and whether they share the restored-pipeline signature:

| ood_case | rank | nearest_train | mahal_dist | is_restored_train |
| -------- | ---- | ------------- | ---------- | ----------------- |
| run_133  | 1    | run_310       | 0.352      | YES               |
|          | 2    | run_44        | 1.087      | YES               |
|          | 3    | run_416       | 1.327      | YES               |
|          | 4    | run_484       | 1.412      | YES               |
|          | 5    | run_249       | 1.650      | YES               |
| run_158  | 1    | run_310       | 0.277      | YES               |
|          | 2    | run_44        | 0.566      | YES               |
|          | 3    | run_416       | 0.804      | YES               |
|          | 4    | run_484       | 0.899      | YES               |
|          | 5    | run_249       | 1.125      | YES               |
| run_203  | 1    | run_249       | 0.221      | YES               |
|          | 2    | run_484       | 0.316      | YES               |
|          | 3    | run_416       | 0.414      | YES               |
|          | 4    | run_184       | 0.608      | YES               |
|          | 5    | run_44        | 0.647      | YES               |
| run_226  | 1    | run_249       | 0.660      | YES               |
|          | 2    | run_184       | 0.688      | YES               |
|          | 3    | run_484       | 0.880      | YES               |
|          | 4    | run_416       | 0.964      | YES               |
|          | 5    | run_44        | 1.206      | YES               |

- run_133: 100% of 5-NN train cases share the restored-pipeline signature.
- run_158: 100% of 5-NN train cases share the restored-pipeline signature.
- run_203: 100% of 5-NN train cases share the restored-pipeline signature.
- run_226: 100% of 5-NN train cases share the restored-pipeline signature.

## Verdict

**EXTRAPOLATIVE** (group-distance test): 4/4 of the OOD-4 test cases are ≥2σ outliers vs the other 46 test cases in at least one of (4D-scalar Mahalanobis, 32-bin histogram chi²) k-NN distance to train.

**Refined diagnosis**: the 4 OOD test cases are NOT geometrically novel — their 5-NN train neighbours are dominated by the 6 'restored' train cases that share an anomalously near-zero `sdf_min` and a `sdf_negative_frac` that is ~10× smaller than typical. The OOD-4 are extrapolative w.r.t. the *bulk* (394) of train, but interpolative w.r.t. a specific 6-case minority pocket (run_44, run_184, run_249, run_310, run_416, run_484).

All 10 cases (4 test + 6 train) where `sdf_min ≈ 0` are exactly the `REQUIRED_RESTORED_CASE_IDS` from `data/loader.py` — public DrivAerML cases that were restored after a previous exclusion. Their `volume_sdf.npy` arrays appear to have been regenerated through a pipeline that did not include the negative-side / inside-body samples that all 394 non-restored cases have.

**Recommendation:**

1. **Highest-leverage fix (data side)**: regenerate `volume_sdf.npy` for the 10 restored cases using the same sampling scheme as the 394 non-restored cases. This is a one-off data fix that should remove the test_vol_p hot-spot on these 4 cases without any architecture change. Flag this as a candidate human-issue or a separate data-team PR.
2. **Per-geometry conditioning experiments are NOT a write-off**: FiLM v3 / SDF-gate v3 / AdaLN-zero are still viable because there are 6 train cases with the matching SDF signature — but 6/400 is a tiny minority pocket, so the conditioning module needs enough capacity to handle a bimodal stat distribution and enough training to learn from those 6. Expect FiLM/SDF-gate to give modest, not dramatic, improvements on the OOD-4 test cases while the data-side bug remains.
3. **Sanity check before any new conditioning experiment**: include `sdf_min`, `sdf_negative_frac`, and `sdf_q05` as model conditioning inputs (cheap, scalar) so the model can use them as an explicit pipeline-mode gate. This collapses the bimodal nuisance into a learnable indicator.
