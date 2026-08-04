# trajectory-judge

How much does an LLM judge miss when an agent reaches the right answer the wrong way?

Outcome-only evaluation is the production default for agents. It cannot see a trajectory that
skipped a required check, acted against what a tool returned, or asserted something no
observation supports — as long as the final answer came out right. This project measures that
blind spot under conditions where the ground truth is known by construction.

Work in progress. Results and design notes to follow.
