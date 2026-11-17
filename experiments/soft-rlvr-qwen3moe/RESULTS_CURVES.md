# Thirteen finetunes in one model: learning curves, alone and co-resident

`Qwen/Qwen3-30B-A3B-Base`, 13 tasks, one adapter a task: a 16-slot cartridge, and a rank-1 LoRA on the attention
projections and every MoE expert. Each adapter is scored by itself and inside the stack of all 13 of its kind, at
snapshots along its training. The runs are `tools/curves.py`; `tools/export_curves.py` wrote `curves/points.csv`,
which holds every point below.

## What was trained

- **Tasks.** Seven trained by RL (math, code, logic, law, casehold, medqa, trivia): a warm start on the base
  model's own correct samples, then GRPO. Six trained by SFT on gold answers (Brainfuck, FinQA, function calling,
  intent, parsing, formula): 10 passes for the cartridge, 3 for the LoRA.
- **Co-residence training.** Every cartridge and every rank-1 LoRA trained beside six filler partners that were
  present with probability 0.3, with the write penalty on off-task prompts. No task's adapter ever trained beside
  another task's adapter, so in the stack every partner is one it has not met.
- **The stack.** Cartridges: all 13 side by side, 208 slots. LoRAs: the 13 rank-1 LoRAs summed (one rank-13
  LoRA), with nothing that picks one of them for a request. At a given point the stack holds every task's
  snapshot at that fraction of its own training.
- **Learning rates** were chosen by sweeps on separate selection problems; the scores below are on each task's
  test problems.

## Scores at each curve's best checkpoint

| Task | Trained by | Plain model | Cartridge alone | LoRA alone | Cartridge in the stack of 13 | LoRA in the stack of 13 |
|---|---|---|---|---|---|---|
| math | RL | 0.33 | 0.76 | 0.75 | 0.74 | 0.31 |
| code | RL | 0.33 | 0.55 | 0.53 | 0.54 | 0.39 |
| logic | RL | 0.29 | 0.36 | 0.50 | 0.36 | 0.28 |
| law | RL | 0.33 | 0.54 | 0.61 | 0.57 | 0.36 |
| casehold | RL | 0.37 | 0.66 | 0.79 | 0.76 | 0.35 |
| medqa | RL | 0.49 | 0.78 | 0.80 | 0.77 | 0.50 |
| trivia | RL | 0.42 | 0.62 | 0.63 | 0.46 | 0.46 |
| Brainfuck | SFT | 0.02 | 0.84 | 0.83 | 0.61 | 0.06 |
| FinQA | SFT | 0.00 | 0.67 | 0.65 | 0.63 | 0.00 |
| function calling | SFT | 0.59 | 0.58 | 0.66 | 0.54 | 0.59 |
| intent | SFT | 0.01 | 0.74 | 0.63 | 0.57 | 0.07 |
| parsing | SFT | 0.00 | 0.54 | 0.66 | 0.28 | 0.52 |
| formula | SFT | 0.04 | 0.11 | 0.15 | 0.07 | 0.04 |
| **mean of 13** | | **0.25** | **0.60** | **0.63** | **0.53** | **0.30** |

Each cell is the best checkpoint of that curve on the test problems: for the stack columns, the task's best score
over the stacks taken along training. The same means at the last checkpoint of every run, where all 13 sit in one
stack: 0.57, 0.60, 0.42 and 0.07.

- Summed, the 13 rank-1 LoRAs lose their scores from the first snapshot on; on 9 of the 13 tasks their best stack
  is the untrained start.
- Cartridges in the stack follow their alone curve on the RL tasks except trivia. On the SFT tasks they peak early
  and fall as training goes on; `RESULTS_SFT.md` section 2 has five and six SFT cartridges together holding their
  scores.

## Figures

- `curves/coresidence_tasks.png`: one panel a task, score against training examples seen, each adapter alone and
  in the stack; the last panel is the mean with every curve cut at its own best checkpoint (`curves/figure_tasks.py`).
- `curves/coresidence_cost.png`: the mean of the 13 against the fraction of training, and for every task the score
  in the stack minus the score alone at the same snapshot (`curves/figure_cost.py`).

      python curves/figure_tasks.py curves/points.csv curves/coresidence_tasks.png
      python curves/figure_cost.py curves/points.csv curves/coresidence_cost.png
