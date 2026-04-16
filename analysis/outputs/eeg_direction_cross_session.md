# EEG Direction Cross-Session Summary

- Ordered session pairs are evaluated independently: train on one session, test on a different unseen session.
- Calibration uses the first `45` seconds of each session separately.
- Windowing: `1.0` s, overlap `0.5`.
- Features: `combined + asymmetry`.
- Model: `RandomForest`.
- Tasks: `LEFT vs ALL`, `RIGHT vs ALL`.

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/eeg_direction_cross_session.csv`

## Per-Pair Results

| train_session | test_session | task | accuracy | macro_f1 | notes |
| --- | --- | --- | --- | --- | --- |
| LR-2-27-26-(01).csv | LR-3-8-26-(02).csv | left_vs_all | 0.8691910499139415 | 0.46500920810313073 | weak separation |
| LR-3-8-26-(03).csv | LR-3-8-26-(02).csv | left_vs_all | 0.8691910499139415 | 0.46500920810313073 | weak separation |
| LR-3-15-26-(04).csv | LR-3-8-26-(02).csv | left_vs_all | 0.8661790017211703 | 0.46414572284989625 | weak separation; false triggers exceed hits |
| LR-2-27-26-(01).csv | LR-3-8-26-(03).csv | left_vs_all | 0.8076416337285902 | 0.4467930029154519 | weak separation |
| LR-3-15-26-(04).csv | LR-3-8-26-(03).csv | left_vs_all | 0.8076416337285902 | 0.4467930029154519 | weak separation |
| LR-3-8-26-(02).csv | LR-3-8-26-(03).csv | left_vs_all | 0.7944664031620553 | 0.44273127753303965 | weak separation; false triggers exceed hits |
| LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | left_vs_all | 0.7896138482023968 | 0.4412202380952381 | weak separation |
| LR-3-8-26-(02).csv | LR-3-15-26-(04).csv | left_vs_all | 0.7896138482023968 | 0.4412202380952381 | weak separation |
| LR-3-8-26-(03).csv | LR-3-15-26-(04).csv | left_vs_all | 0.7896138482023968 | 0.4412202380952381 | weak separation |
| LR-3-8-26-(02).csv | LR-2-27-26-(01).csv | left_vs_all | 0.7694994179278231 | 0.43978260869565217 | weak separation |
| LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | left_vs_all | 0.7683352735739232 | 0.43449637919684003 | weak separation |
| LR-3-8-26-(03).csv | LR-2-27-26-(01).csv | left_vs_all | 0.7683352735739232 | 0.43449637919684003 | weak separation |
| LR-2-27-26-(01).csv | LR-3-8-26-(02).csv | right_vs_all | 0.8730636833046471 | 0.46611532276590856 | weak separation |
| LR-3-15-26-(04).csv | LR-3-8-26-(02).csv | right_vs_all | 0.8730636833046471 | 0.46611532276590856 | weak separation |
| LR-3-8-26-(03).csv | LR-3-8-26-(02).csv | right_vs_all | 0.8730636833046471 | 0.46611532276590856 | weak separation |
| LR-2-27-26-(01).csv | LR-3-8-26-(03).csv | right_vs_all | 0.8247694334650857 | 0.451985559566787 | weak separation |
| LR-3-15-26-(04).csv | LR-3-8-26-(03).csv | right_vs_all | 0.8247694334650857 | 0.451985559566787 | weak separation |
| LR-3-8-26-(02).csv | LR-3-8-26-(03).csv | right_vs_all | 0.8247694334650857 | 0.451985559566787 | weak separation |
| LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | right_vs_all | 0.8079161816065192 | 0.4468770122343851 | weak separation |
| LR-3-8-26-(02).csv | LR-2-27-26-(01).csv | right_vs_all | 0.8079161816065192 | 0.4468770122343851 | weak separation |
| LR-3-8-26-(03).csv | LR-2-27-26-(01).csv | right_vs_all | 0.8079161816065192 | 0.4468770122343851 | weak separation |
| LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | right_vs_all | 0.8015978695073236 | 0.4449371766444937 | weak separation |
| LR-3-8-26-(02).csv | LR-3-15-26-(04).csv | right_vs_all | 0.8015978695073236 | 0.4449371766444937 | weak separation |
| LR-3-8-26-(03).csv | LR-3-15-26-(04).csv | right_vs_all | 0.8015978695073236 | 0.4449371766444937 | weak separation |

## Task Summary

| task | accuracy_mean | accuracy_variance | macro_f1_mean | macro_f1_variance |
| --- | --- | --- | --- | --- |
| right_vs_all | 0.8268367919708939 | 0.0007840480393141907 | 0.4524787678028936 | 6.861353706831564e-05 |
| left_vs_all | 0.8074435234875957 | 0.0013913510101523825 | 0.44690979198292896 | 0.00011881981241206154 |

## Session-Pair Summary

| train_session | test_session | accuracy | macro_f1 |
| --- | --- | --- | --- |
| LR-2-27-26-(01).csv | LR-3-8-26-(02).csv | 0.8711273666092942 | 0.4655622654345196 |
| LR-3-8-26-(03).csv | LR-3-8-26-(02).csv | 0.8711273666092942 | 0.4655622654345196 |
| LR-3-15-26-(04).csv | LR-3-8-26-(02).csv | 0.8696213425129087 | 0.4651305228079024 |
| LR-2-27-26-(01).csv | LR-3-8-26-(03).csv | 0.8162055335968379 | 0.44938928124111943 |
| LR-3-15-26-(04).csv | LR-3-8-26-(03).csv | 0.8162055335968379 | 0.44938928124111943 |
| LR-3-8-26-(02).csv | LR-3-8-26-(03).csv | 0.8096179183135706 | 0.4473584185499133 |
| LR-3-8-26-(02).csv | LR-2-27-26-(01).csv | 0.7887077997671712 | 0.4433298104650186 |
| LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.7956058588548602 | 0.4430787073698659 |
| LR-3-8-26-(02).csv | LR-3-15-26-(04).csv | 0.7956058588548602 | 0.4430787073698659 |
| LR-3-8-26-(03).csv | LR-3-15-26-(04).csv | 0.7956058588548602 | 0.4430787073698659 |
| LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.7881257275902211 | 0.4406866957156126 |
| LR-3-8-26-(03).csv | LR-2-27-26-(01).csv | 0.7881257275902211 | 0.4406866957156126 |

## Recommendation

- Best generalizing task: `right_vs_all`
- left_vs_all: mean macro-F1 `0.447`, variance `0.0001`
- right_vs_all: mean macro-F1 `0.452`, variance `0.0001`
- Recommended direction model: Use RIGHT vs ALL as the more stable cross-session direction model.