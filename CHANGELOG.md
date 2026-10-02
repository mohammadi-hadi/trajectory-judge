# Changelog

## 0.2.0 (2 October 2026)

The camera-ready release for the paper *trajectory-judge: What Outcome-Only LLM Judges Miss on
Agent Trajectories* (NeurIPS 2026 workshop "Who Verifies the Agents? Toward Reliable Agent
Development"). The v0.1.0 verdicts and every table built from them are unchanged.

### Added

- Paired analysis (`analysis/paired_ci.py`, `analysis/paired.json`): each judge's verdict on a
  fault is compared with its verdict on the clean run the fault was made from.
- View-by-task ablation (`results/ablation/`): four judges that cross what the judge is shown
  with what it is asked, under one response schema, plus the outcome judge with its own schema.
  2,205 verdicts over the 400 trajectories and the 41 clean parents they lacked, with the raw
  model responses, the engine log, and the predictions written before the runs.
- The same two prompts on the 60 agent-driven episodes (`results/ablation/organic/`).
- `analysis/` and `data/`: the scripts and inputs that rebuild every number in the paper, with
  `make numbers` and a CI check that the committed outputs match the committed verdicts.
- `run` options: `--with-parents`, `--first`, `--source`, `--keep-responses`.
- Golden hashes of every judge prompt (`tests/test_prompts.py`).

### Changed

- A failed model call is no longer stored as a clean verdict. It is left unjudged, the run
  stops after three failures in a row, and the exit code says so.
- "Customer-visible outcome" is now "environment outcome" in code, tables and docs. It is the
  refund or escalation the episode ends in, and does not include the reply text.
- The README follows the paper's corrected claims: recall is read against clean parents, the
  two judges differ in instruction and schema as well as in view, and step localisation is
  given with out-of-range steps counted as misses (0.844 of detected faults, 0.723 of all).

### Notes

- The v0.1.0 verdicts came from Ollama 0.30.11 and the ablation from Ollama 0.33.3. The
  outcome judge's verdicts differ between the two engines; see `data/PROVENANCE.md`.

## 0.1.0 (4 August 2026)

First release: the environment, oracle, fault injector, five judges, and the raw verdicts of
the main comparison.
