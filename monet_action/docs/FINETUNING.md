# How much of Monet to train: frozen, LoRA, or full fine-tuning

**Short answer.** The action expert (DiT), the conditioner and the state / TOOL_RESULT projectors are always
trained **in full, from scratch**. The open question is only how much of **Monet-7B**, the reasoning model,
to adapt.

The proposal's primary jointly trained model D2 uses LoRA. LoRA does train the reasoning: the flow-matching
loss is back-propagated through all K=10 self-fed latent steps, so the adapters change every latent step.
But D2's LoRA is deliberately small: about 2.9M parameters, attention only, in the last 8 of 28 layers. It
may lack the capacity to adapt Monet's reasoning to manipulation.

Training the whole of Monet is not possible on one 96 GB GPU. It is also risky for this research question
(see below). This repository therefore treats "how much of Monet to train" as an experiment: a **capacity
ladder**. Every level is paired with a matched K=0 control.

## 1. What is trained in each run

| level | runs (K=10 / matched K=0) | Monet weights trained | trained Monet params | lr (Monet) |
|---|---|---|---|---|
| frozen | D1 `d1_k10` / D0 `d0_k0` | none | 0 | — |
| LoRA (pre-registered D2) | `d2_lora` / `d0_k0_lora` | LoRA r=16, α=32 on q/k/v/o, layers 20–27 | ~2.9M | 5e-5 |
| wide LoRA | `d2_lora_wide` / `d0_k0_lora_wide` | LoRA r=64, α=128 on q/k/v/o **and** gate/up/down, all 28 layers | ~161M | 5e-5 |
| full FT, 4 layers | `d2_ft4` / `d0_k0_ft4` | every weight of decoder layers 24–27 (fp32) | ~0.93B | 1e-5 |
| full FT, 8 layers | `d2_ft8` / `d0_k0_ft8` | every weight of decoder layers 20–27, the layers D2 adapts with LoRA | ~1.86B | 1e-5 |
| full LLM | not provided | all 28 decoder layers | ~6.5B | — |

The structure is the same at every adapted level:

* stage A is the frozen run's first 10k steps (D1, or D0 for the controls);
* stage B is 10k joint steps from that checkpoint, so every run gets 20k head updates;
* same data, split, seeds, effective batch (16), DiT and evaluation protocol.

The vision tower, the embeddings and the final norm stay frozen everywhere. Fine-tuned layers are kept in
fp32: bf16 weights would silently drop small AdamW updates. Their matmuls still run in bf16 under autocast.

Parameter counts per Monet-7B decoder layer: attention 29.4M, MLP 203.7M, so 233.1M per layer and 6.53B for
all 28 layers.

## 2. Why not simply fully fine-tune Monet?

1. **It does not fit.** Each trained parameter costs about 16 extra bytes of GPU memory over a frozen bf16
   one: fp32 weight (+2), fp32 gradient (4), AdamW moments (8) and the bf16 copy autocast keeps for the
   backward pass (2). This comes on top of the 16.6 GB of frozen weights and the activations.

   | level | extra memory (approx.) | in a 50% share (~47 GB)? |
   |---|---|---|
   | LoRA (D2) | ~0.05 GB | yes (preflight-measured) |
   | wide LoRA | ~2.6 GB + adapter activations | probably; profiled before training |
   | full FT, 4 layers | ~15 GB | maybe, at a small micro-batch; profiled |
   | full FT, 8 layers | ~30 GB | no: needs the whole 96 GB card |
   | full LLM | ~104 GB | no: more than the whole GPU before any activation |

   The whole LLM would need 8-bit or paged optimizer states, CPU offload, or several GPUs. None of that is
   part of Phase 1.
2. **It can destroy what we want to test.** LIBERO-Spatial has 432 demonstrations, about 53k frames. Fully
   fine-tuning a 7B model on that can overwrite Monet's pretrained latent reasoning. A gain would then show
   "a fine-tuned 7B VLM helps", not "Monet's latent reasoning helps", which is the research question (H2).
3. **It confounds capacity with mechanism.** More trainable capacity improves the K=0 model too. Only
   a K=10 vs K=0 comparison *at the same level of adaptation* isolates the latent steps. That is why every
   level has a matched control.
4. **The evidence does not demand it.** For a 7B VLA, OpenVLA found LoRA matched full fine-tuning while
   training about 1.4% of the parameters ([Kim et al. 2024](https://arxiv.org/pdf/2406.09246)). Its LoRA
   covered all linear layers, not only attention. OpenVLA-OFT reached 97.1% average LIBERO success
   ([Kim, Finn, Liang 2025](https://arxiv.org/abs/2502.19645v1)). Community guides to its released recipe
   use LoRA too. Analyses of LoRA vs full fine-tuning ([Thinking Machines, "LoRA Without
   Regret"](https://thinkingmachines.ai/blog/lora/)) find that LoRA needs to cover the MLP layers to match
   full fine-tuning. That is exactly what D2's attention-only LoRA does not do, and what the wide-LoRA level
   tests.

Your concern is therefore reasonable for D2 as configured. The cheap, direct test is the wide LoRA, not full
fine-tuning.

## 3. How to run the ladder

The ladder needs the D0 and D1 stage-A checkpoints (`runs/d0_k0/ckpt_0010000`,
`runs/d1_k10/ckpt_0010000`), which `train_all.sh` keeps.

```bash
bash scripts/train_capacity.sh lora_wide
```
```bash
bash scripts/train_capacity.sh ft4
```
```bash
MONET_GPU_RESERVE=0.9 bash scripts/train_capacity.sh ft8
```
The last one needs the card to yourself, and the holder stopped first (`bash scripts/gpu_hold.sh stop`).
You can also append the ladder to the main sequence:
```bash
CAPACITY="lora_wide ft4" bash scripts/train_all.sh
```

* **Batch size.** Each level profiles its memory once (`results/preflight/memory_profile_d2_<level>.json`)
  and picks the largest micro-batch that fits and divides 16. The rest comes from gradient accumulation.
  Its matched K=0 control uses the same micro-batch.
* **Disk.** Full fine-tuning checkpoints are large. FT4 is ~11 GB per checkpoint (weights + optimizer
  state), FT8 ~22 GB. These configs keep the last 2 checkpoints plus step 10k.
* **Evaluation.** `scripts/evaluate_all.sh` evaluates every ladder run that exists. The report
  (`results/phase1_report.md`, section 4d) shows the K=10 − K=0 gap at each level, with paired statistics.

## 4. Decision rule (fixed before any result)

1. **Run the pre-registered core first:** D0, D1, D2 and D0-LoRA. Their conclusions stand on their own.
2. **Run wide LoRA** (`lora_wide`) as the first ladder step. It is cheap, fits the 50% share and answers
   "is D2's LoRA too small?" directly.
3. **Run full FT of 4 layers** (`ft4`) only if one of these holds on development init states:
   * `d2_lora_wide` beats `d2_lora` by more than its CI, so more capacity helped; or
   * D2 under-fits: its stage-B training loss ends within 5% of D1's.
4. **Run full FT of 8 layers** (`ft8`) only if `ft4` beats `lora_wide`, and the whole card is available.
5. **Learning rate first.** Before blaming capacity, check the learning rate: LoRA's best rate is typically
   several times the full fine-tuning rate, and D2's 5e-5 is conservative. A short sweep (1e-4, 2e-4 on
   development states) is cheaper than full fine-tuning.

**Reading the outcome:**

| pattern across levels | interpretation |
|---|---|
| the K=10 − K=0 gap is about the same at every level | the latent benefit does not depend on adapting Monet |
| the gap grows with capacity | the reasoning must be adapted to help manipulation: supports training the reasoner |
| both improve and the gap shrinks to 0 | the K=10 advantage was a feature-adaptation shortcut; with enough capacity plain VL features suffice (against H2) |
| both get worse with more capacity | overfitting / forgetting on 432 demos: stay at LoRA |

The ladder is **exploratory**: it is registered in `RESEARCH_MEMO.md` before any result, but outside the two
primary contrasts, so it carries no multiplicity-corrected claim.
