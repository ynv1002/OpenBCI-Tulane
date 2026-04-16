# EEG Direction Base-Up Calibrated Summary

- Sessions are evaluated independently.
- Calibration uses the first `45` seconds of BASELINE/REST windows.
- Remaining data is split chronologically: first `70%` train, last `30%` test.
- Windowing: `1.0` s, overlap `0.5`.
- Features: `combined + asymmetry`.
- Model: `RandomForest`.

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/eeg_direction_base_up_calibrated.csv`

## Per-Run Results

| session_name | task | accuracy | macro_f1 | precision | recall | notes |
| --- | --- | --- | --- | --- | --- | --- |
| LR-3-15-26-(04).csv | left_vs_all | 0.9379157427937915 | 0.904414436235921 | 0.8893499912633234 | 0.9221769609700644 | strong separation |
| LR-2-27-26-(01).csv | left_vs_all | 0.8023255813953488 | 0.6188627871274224 | 0.714986123959297 | 0.6027227722772277 | mixed performance |
| LR-3-8-26-(03).csv | left_vs_all | 0.9868421052631579 | 0.4966887417218543 | 0.5 | 0.4934210526315789 | weak separation; false triggers exceed hits; test missing LEFT |
| LR-3-8-26-(02).csv | left_vs_all | 0.7747489239598279 | 0.43654001616814875 | 0.40358744394618834 | 0.4753521126760563 | weak separation; false triggers exceed hits |
| LR-3-15-26-(04).csv | left_vs_right | 0.9028571428571428 | 0.9028063641412656 | 0.9040728129910949 | 0.9030172413793103 | strong separation |
| LR-3-8-26-(02).csv | left_vs_right | 0.5887096774193549 | 0.5817460317460317 | 0.5885164197446576 | 0.5844570386294052 | mixed performance; directional bias |
| LR-2-27-26-(01).csv | left_vs_right | 0.49572649572649574 | 0.4549545992893802 | 0.512768817204301 | 0.5083430913348946 | weak separation; directional bias |
| LR-3-8-26-(03).csv | left_vs_right | 0.6666666666666666 | 0.4 | 0.5 | 0.3333333333333333 | weak separation; test missing LEFT |
| LR-3-15-26-(04).csv | right_vs_all | 0.9312638580931264 | 0.894598364054431 | 0.8813704933310036 | 0.9099517906336088 | strong separation |
| LR-3-8-26-(03).csv | right_vs_all | 0.9736842105263158 | 0.6932735426008969 | 0.7410714285714286 | 0.6621621621621622 | mixed performance |
| LR-2-27-26-(01).csv | right_vs_all | 0.7596899224806202 | 0.5216507177033493 | 0.6222222222222222 | 0.5370724806524091 | mixed performance; false triggers exceed hits |
| LR-3-8-26-(02).csv | right_vs_all | 0.8192252510760402 | 0.4503154574132492 | 0.413768115942029 | 0.4939446366782007 | weak separation; false triggers exceed hits |

## Task Averages

| task | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | precision_mean | precision_std | recall_mean | recall_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| right_vs_all | 0.8709658105440257 | 0.09873589855689953 | 0.6399595204429815 | 0.1980279181349112 | 0.6646080650166708 | 0.19794766649798956 | 0.6507827675315951 | 0.1869280626244078 |
| left_vs_all | 0.8754580883530315 | 0.10295315631873479 | 0.6141264953133366 | 0.2078606140734859 | 0.6269808897922021 | 0.21803000150357288 | 0.6234182246387319 | 0.20696853426253217 |
| left_vs_right | 0.663489995667415 | 0.17420620951628393 | 0.5848767487941694 | 0.2252022934168561 | 0.6263395124850134 | 0.18923208659909505 | 0.5822876761692358 | 0.2382703132833982 |

## Session Averages

| session_name | macro_f1 | accuracy |
| --- | --- | --- |
| LR-3-15-26-(04).csv | 0.9006063881438725 | 0.924012247914687 |
| LR-2-27-26-(01).csv | 0.5318227013733839 | 0.6859139998674882 |
| LR-3-8-26-(03).csv | 0.5299874281075837 | 0.8757309941520468 |
| LR-3-8-26-(02).csv | 0.4895338351091432 | 0.7275612841517409 |

## Summary

- Best performing task: `right_vs_all`
- `right_vs_all`: mean macro-F1 `0.640`, std `0.198`
- `left_vs_all`: mean macro-F1 `0.614`, std `0.208`
- `left_vs_right`: mean macro-F1 `0.585`, std `0.225`
- Recommended direction model: Use RIGHT vs ALL as the most reliable calibrated direction detector.