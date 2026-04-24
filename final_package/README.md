# Final Package

This folder is the professor-facing package for the `clean-file` branch.

It is organized so you can link directly to specific folders in GitHub without needing to explain the rest of the repo first.

## Recommended GitHub Links

If you want the cleanest set of links for the Word document, use these folders:

- `final_package/report/`
  - write-up, dataset table, modeling table, evidence map
- `final_package/results/`
  - offline results, failed iterations, and the stepwise benchmark summaries
- `final_package/gui_game_code/`
  - the hybrid GUI/game code that runs baseline, guided collection, adaptation, and gameplay
- `final_package/live_run_examples/`
  - complete logged example runs from the hybrid system

Optional supporting links:

- `final_package/analysis_code/`
  - minimal offline scripts behind the reported analyses
- `final_package/models/`
  - runtime artifacts used by the current hybrid demo

## Reading Order

For someone opening the repo fresh, the best order is:

1. `report/`
2. `results/`
3. `gui_game_code/`
4. `live_run_examples/`

## Package Structure

- `report/`
  - main report draft and supporting tables
- `results/`
  - curated benchmark summaries and comparison writeups
- `gui_game_code/`
  - current tracking-game GUI/runtime stack
- `live_run_examples/`
  - full logged runs that show the live/replay evidence structure
- `analysis_code/`
  - narrower offline benchmark code behind the results
- `models/`
  - saved jaw and hand artifacts used by the runtime
