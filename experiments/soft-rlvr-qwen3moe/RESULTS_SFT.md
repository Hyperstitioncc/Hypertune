# Supervised fine-tuning: cartridges against LoRA

A 16-slot key/value cartridge (786,432 trained values) and a rank-1 LoRA (52.7M trained values) are trained on the
same examples of a frozen `Qwen/Qwen3-30B-A3B-Base`: 1,000 examples a domain (700 for function calling). Answers
are greedy and graded by each benchmark's own check (answer match, program output, or parsed call). One seed.
`runs_sft.sh` holds the commands. The 13-task learning curves are in `RESULTS_CURVES.md`.

## 1. One adapter per domain

Cartridge: 16 slots, 10 passes, rate 1e-2. LoRA: rank 1 on the attention projections and every MoE expert, 3 passes,
rate 1e-4.

| Domain | Data | Test problems | Plain model | Cartridge | LoRA rank 1 |
|---|---|---|---|---|---|
| Brainfuck | generated program-writing tasks, about 30 task types | 699 | 0.02 | 0.68 | 0.68 |
| Intent routing | Banking77 | 500 | 0.01 | 0.77 | 0.80 |
| Finance QA | FinQA | 500 | 0.00 | 0.68 | 0.71 |
| Function calling | Hermes single-turn | 275 | 0.58 | 0.67 | 0.65 |
| Parsing | TOPv2 | 500 | 0.00 | 0.50 | 0.60 |

- The cartridge is within about 0.03 of the LoRA on four domains and 0.10 behind on parsing, with 67 times fewer
  trained values.
- Brainfuck's 699 test problems are 452 on task statements that are also in the training set, each with new hidden
  test inputs, and 247 on statements that are not (the 48 of the builder's test split and the 199 of
  `bf_unseen/`). Every model's result on every problem of `bf_unseen/` is in `bf_unseen/scores.jsonl`.
- The LoRA scores for finance and function calling come from the first version of the trainer (function calling
  on 100 problems); everything else in the table is from the library.
- The cartridge needs more passes than the LoRA, whose dev score is flat after 2 to 3. With 64 slots in place of
  16 it gains 0.06 on parsing.
- Formula from name (SMolInstruct) is learned by the LoRA (0.33) and barely by the cartridge (0.13; plain model
  0.04). MedQA, text-to-SQL and molecule generation showed no gain for either adapter at 1,000 examples.

## 2. Co-residence: cartridges trained separately, loaded together

Each domain's cartridge is trained by itself, never with another domain's cartridge, and then scored on its own
domain with the other domains' cartridges loaded beside it. Two things make that work: a penalty on what the
cartridge writes on other tasks' prompts (strength 1.0; eight public text sources and six filler tasks supply
those prompts), and six filler cartridges present during training with probability 0.3.

| Domain | Alone | Five together | Six together (formula also loaded) |
|---|---|---|---|
| Brainfuck | 0.84 | 0.83 | 0.83 |
| Finance QA | 0.65 | 0.64 | 0.64 |
| Function calling | 0.63 | 0.65 | 0.64 |
| Intent routing | 0.66 | 0.63 | 0.62 |
| Parsing | 0.45 | 0.43 | 0.43 |

Brainfuck here is scored on the builder's 500 test problems.

- Loading the others costs 0.00 to 0.03 a domain with five and 0.00 to 0.04 with six.
- Training for co-residence costs 0.02 to 0.05 alone against the plain cartridge of section 1, and 0.11 on intent
  routing.
- Both ingredients are needed. Cartridges trained without the penalty fall to zero on four of five domains when
  the others are loaded; with the penalty but no partners during training, Brainfuck, intent routing and parsing
  still collapse.
- Stacks hold to about ten cartridges. With twelve loaded intent routing and parsing fall by half, and with
  sixteen they are near zero; `RESULTS_CURVES.md` has thirteen tasks together.

Summed LoRAs on the same domains, alone → six together (first version of the trainer, 100 test problems, LoRAs
trained with no penalty and no partners):

| LoRA | Brainfuck | Finance QA | Function calling | Intent routing | Parsing | Formula |
|---|---|---|---|---|---|---|
| Rank 1 | 0.82 → 0.83 | 0.73 → 0.72 | 0.65 → 0.64 | 0.80 → 0.76 | 0.57 → 0.42 | 0.37 → 0.15 |
| Rank 8 | 0.84 → 0.70 | 0.72 → 0.70 | 0.66 → 0.60 | 0.85 → 0.71 | 0.67 → 0.17 | 0.46 → 0.04 |
| Rank 32 | 0.84 → 0.45 | 0.76 → 0.70 | 0.66 → 0.21 | 0.84 → 0.72 | 0.64 → 0.01 | 0.42 → 0.07 |

- Six rank-1 LoRAs summed still score above six cartridges stacked, because they start higher; they lose more
  from the sum on parsing and formula than the cartridges lose from the stack.
- Summed LoRAs get worse as the rank grows, and with thirteen tasks the rank-1 sum collapses
  (`RESULTS_CURVES.md`).

## Recipe

- **Cartridge:** 16 slots × 48 layers × 4 key/value heads × 128 × (keys + values), all slots at rotary position 0,
  placed between the prompt and its last token, started from the keys and values of real task tokens. AdamW, rate
  1e-2, cosine decay, batch 16, 10 passes.
- **LoRA:** rank 1 on attention q/k/v/o and every expert's gate/up/down, rate 1e-4, batch 16, 3 passes.
- **Data and graders:** `envs/domains/` (seed 0). `runs_sft.sh DATA OUT` builds the data, trains both adapters on
  every domain, trains the co-residing cartridges and scores the stacks.
- **Brainfuck's test set** is the builder's 500 test problems and `bf_unseen/test.jsonl`. To score an adapter on
  the second part, put that file beside the trained set's `train.jsonl` and `dev.jsonl` in a data folder and run
  `python -m shared.cli bank` and `eval` on it as `runs_sft.sh` does. `bf_unseen/prompt_head.txt` is the
  2,250-token prompt (the language description and one solved training problem of every task type) of the prompt
  baseline.
