# Future Jaw Click Collection Protocol

This plan is aimed at live click detection rather than coarse hold-vs-repeated blocks.
Marker edges still represent cue timing, not exact physiological onset truth, so the design favors isolated tasks and long inactive gaps.

## Proposed Blocks

|   block_order | task_name                |   marker_start |   marker_end |   repetitions |   trial_duration_sec |   rest_after_sec | experimenter_prompt                                                | why_it_helps                                                                         |
|--------------:|:-------------------------|---------------:|-------------:|--------------:|---------------------:|-----------------:|:-------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
|             1 | baseline_quiet           |              0 |            0 |             1 |                 20   |              0   | Relax jaw and stay still.                                          | Improves inactive calibration and makes quiet false-trigger behavior measurable.     |
|             2 | single_click             |             11 |           12 |            20 |                  2   |              4   | One quick clench, then fully relax.                                | Provides the cleanest onset and offset examples for tap-style control.               |
|             3 | short_hold_release       |             21 |           22 |            15 |                  1.5 |              4   | Clench briefly, hold about one second, then release.               | Strengthens active-hold labels while keeping onset and offset visually clean.        |
|             4 | double_click             |             31 |           32 |            15 |                  2.5 |              4.5 | Two quick clenches with a clear gap between them.                  | Tests minimum separation and whether the detector splits consecutive taps correctly. |
|             5 | paced_burst_slow         |             41 |           42 |            10 |                  4   |              5   | Repeat clicks at a slow steady tempo, about two clicks per second. | Gives labeled burst structure without the ambiguity of unscripted repeated blocks.   |
|             6 | paced_burst_fast         |             51 |           52 |            10 |                  3   |              5   | Repeat clicks at a faster tempo, about three clicks per second.    | Stress-tests cooldown and burst handling for game-like rapid taps.                   |
|             7 | random_wait_single_click |             61 |           62 |            15 |                  2   |              3   | Wait for the cue, then make one click after a random delay.        | Reduces anticipation effects and improves onset realism for future live control.     |

## Practical Notes

- Use a short audio cue or spoken cue at each trial start so the experimenter and participant share timing.
- Keep long inactive gaps after each trial so replay evaluation can distinguish missed clicks from detector chatter.
- Prefer dedicated marker codes per task if the acquisition path supports them.
- If the current marker stack only supports `1-4`, record each task in a separate file so the file identity carries task meaning.
- Log cue timestamps separately if possible; even a simple text log will help compare cue timing to physiological onset later.