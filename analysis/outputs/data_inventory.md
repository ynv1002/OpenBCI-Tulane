# Data Inventory

Session order uses the embedded filename date first and the run index in parentheses second.

| family | session_rank | filename | parsed_date | shape | marker_column_exists | sample_rate_used_hz | usable_for_modeling | top_issue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jaw | 1 | HR-2-27-26-(01).csv | 2026-02-27 | 127218x24 | True | 249.26 | True | unexpected marker transition 1->3 between event 25 and 26 |
| jaw | 2 | HR-3-8-26-(02).csv | 2026-03-08 | 180376x24 | True | 249.186 | True | movement 1 produced 18 interval(s); protocol expected about 10 |
| jaw | 3 | HR-3-15-26-(03).csv | 2026-03-15 | 104968x24 | True | 249.186 | True | unexpected marker transition 2->2 between event 34 and 35 |
| left_right | 1 | LR-2-27-26-(01).csv | 2026-02-27 | 126770x24 | True | 249.186 | True | movement 1 produced 12 interval(s); protocol expected about 10 |
| left_right | 2 | LR-3-8-26-(02).csv | 2026-03-08 | 322569x24 | True | 249.186 | True | movement 1 produced 27 interval(s); protocol expected about 10 |
| left_right | 3 | LR-3-8-26-(03).csv | 2026-03-08 | 113519x24 | True | 249.186 | True | movement 1 produced 9 interval(s); protocol expected about 10 |
| left_right | 4 | LR-3-15-26-(04).csv | 2026-03-15 | 211799x24 | True | 249.246 | True | unexpected marker transition 2->2 between event 74 and 75 |