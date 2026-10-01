# View-by-task ablation: what is predicted and how it will be read

Written and pushed before any of these runs started. The commit that adds this file is the
timestamp; nothing below was changed after the first verdict existed.

## Why this run exists

The published outcome judge and step judge differ in what they are shown (the goal and the final
reply, or the whole trajectory) and in what they are asked to do (decide whether the answer
resolves the request, or check each step against the procedure). They also use different output
schemas. Reviewers rightly pointed out that the published comparison therefore contrasts two
configurations, not two views. This run crosses the two factors under one schema.

## Setup

- Model `qwen2.5:14b` (blob `sha256:2049f5674b1e`), temperature 0, seed 7, `num_ctx` 8192.
- One response schema for every cell: the step judge's (`STEP_SCHEMA`).
- Ollama 0.33.3 on the same laptop as the v0.1.0 runs, one session, `OLLAMA_NUM_PARALLEL=1`,
  flash attention off, as in August. The v0.1.0 verdicts came from Ollama 0.30.11, so no cell
  here is compared with a v0.1.0 verdict except as a reproducibility check.
- Judged set: the 400 benchmark trajectories plus the 41 clean parents the benchmark lacks
  (`run --with-parents`), 441 in all. `premature_stop` and `unsupported_claim` are judged first.
- Cells (judge ids), prompts pinned by hash in `tests/test_prompts.py`:

  | | outcome task | process task |
  |---|---|---|
  | outcome view (goal + reply) | A′ `outview-outtask` (published outcome prompt) | B `outview-proctask` |
  | step view (full trajectory) | C `stepview-outtask` | D′ `stepview-proctask` (published step prompt) |

- The outcome prompt's caveat ("not being able to see the steps is not evidence of a failure")
  is treated as part of the outcome view, so it appears in A′ and B and in neither full-view cell.
- Also run, and reported descriptively only: A′ and D′ on the 60 agent episodes in
  `results/agent` (`run --source results/agent`).

## Estimands and intervals

- Paired discrimination: the share of faults a cell flags minus the share of their clean parents
  it flags, per fault type, pooled over the four fault types that leave the final reply unchanged
  (`wrong_tool`, `hallucinated_argument`, `skipped_precondition`, `ignored_observation`; 200
  faults) and over the two that change it (`premature_stop`, `unsupported_claim`; 100).
- Intervals: instance-clustered bootstrap, B = 10,000, for pooled rows and every contrast, since
  one clean parent serves several faults. Exact McNemar tests only for single-type rows.
- By construction, not a finding: in the outcome-view cells (A′, B) a reply-unchanged fault and its
  parent produce the same prompt, so under deterministic decoding their verdicts agree and the
  paired discrimination is zero. The count of identical verdicts is reported as a determinism check.

## Confirmatory contrasts (Holm-corrected, α = 0.05)

1. View under the process task, where not fixed by construction: D′ minus B on the 100
   reply-changing faults. Predicted positive.
2. Task under the full view, on the 200 reply-unchanged faults: D′ minus C. Predicted positive.
   C's task says to report faulty only when the answer gives a reason to, so a large value is read
   as the judge following that instruction, not as the evidence being insufficient.
3. Task under the full view, on `unsupported_claim` (50): C minus D′. No direction predicted.
4. Task under the outcome view, false alarms: B's flag rate minus A′'s on the 141 clean runs.
   No direction predicted.
5. The published comparison repeated in one session: D′ minus A′ pooled over all 300 faults.
   Predicted positive.

Everything else is reported with intervals and no test. Any analysis chosen after seeing these
verdicts will be labelled post hoc.

## Decision rules fixed now

- If B flags at least 80% or at most 2% of the 141 clean runs, its contrasts are reported but not
  interpreted: the cell would then measure the prompt's internal tension rather than the view.
- Reading C by fault group: on the 125 reply-unchanged faults whose environment outcome survived,
  the outcome task's correct answer is "clean", so C there measures compliance; on the 75 whose
  outcome broke, the reading is conditional (their reply is the clean run's reply, a property of
  the injector); `premature_stop` and `unsupported_claim` carry the task contrast.
- If neither C nor D′ flags most `unsupported_claim` faults, the conclusion is that these
  instructions did not resolve it for this model; capacity and prompt are not separated.
- `premature_stop` localisation, read from D′'s raw responses, is reported under two conventions
  declared here: step 3, the last executed step (the label), and step 4, the missing terminal
  action. Headline numbers keep the label convention.
- D′ against the v0.1.0 step verdicts: per-item agreement on `faulty`, failure type, failure step,
  confidence and stored rationale, reported as rates. Disagreement is attributed to the engine.
- A′ against the v0.1.0 outcome verdicts: agreement on `faulty`, reported. No rate predicted.
- Failed calls are not stored and are retried; unparseable answers are stored, counted and
  reported; a cell with more than 2% unparseable answers is flagged in the paper.
- Expected counts: 441 verdicts per cell, 60 per judge on the agent episodes.

## Not run

A separate reply-checking judge was considered and dropped before any run: its prompt would
have been tuned on the same trajectories it is scored on.
