# NanoPoor
NanoGPT-speedrunning for the poor T4 enjoyers

[Colab Notebook](https://colab.research.google.com/drive/1x87U-mCZCt7Kwc5-HGPOR1NVCOYAN1dr?usp=sharing) 

Inspired by [Modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt) and my goat [Jonas Geiping (Cramming)](https://arxiv.org/pdf/2212.14034), I trained a custom GPT I've been working on over at [Dagonet](https://github.com/BambooML/Dagonet), got to the 3.28 val loss on a single T4.

**Important! Note/Future-bugifx**

As [@main_horse](https://x.com/main_horse/status/1907238044434104633) pointed out, I wrote a method that had the DSMoE class send the current tok to all experts, then apply router weights, so it removed the hard selection of the router, and made it more of a soft weighing instead, the hard routing is loss ~0.1 lower, or about 10 steps faster, but wallclock time per step is 2x longer and init was 8x longer, working on GEMMs

**caveats:**
 - Less than the 120M from main speedrun for stability
 - was just a 1B subset of finewebedu10b, not filtered or anything I just processed that much at this time, will probably fix this later

## Runs

| Ranking  | Time - date | Data | Person | Description | log |
| -------- | ----------- | ---- | ------ | ----------- | --- |
| 1        | 7.09m - 4/5/25 | ~3.27M tok (1024 * 8 * 4 * 100) | Vatsa  | now GPT-2 tokenizer shrunk vocab_size, and also shrunk head_lm and n_experts for stability, less params, now at ~73m | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/shrunk_run.txt) |
| 2        | 11.69m - 4/4/25 | ~3.93M tok (1024 * 8 * 4 * 120) | Vatsa  | lr tuning (5e-4) | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/lr_test_runs.txt) |
| 3        | 14.86m - 4/2/25 | ~5.21M tok (1024 * 8 * 4 * 160) | Vatsa  | 3x lr, removed ckpt saves every step, less printing | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/tweaks_run_nosave.txt) |
| 4        | 15.04m - 4/1/25 | ~3.89M tok (1024 * 5 * 4 * 190) | Vatsa  | Used Muon instead | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/Muon_run.txt) |
| 5        | 37.17m - 4/1/25 | ~6.14M tok (1024 * 5 * 4 * 300) | Vatsa  | Added PSGD | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/GPT4-tok-run.txt) |
| 6        | 70.61m - 3/31/25 | ~14M tok (1024 * 6 * 4 * 570) | Vatsa  | First Run, has DS-MoE, MLA+NSA hybrid, Rope, etc | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/PSGD_run.txt) |

## Unofficial Runs

| Ranking  | Time - date | Data | Person | Description | log |
| -------- | ----------- | ---- | ------ | ----------- | --- |
| 1st      | 7.63m - 4/1/25 | ~6.96M tok (1024 * 10 * 4 * 170) | Vatsa  | Used an A100 with (15.04m - 4/1/25) run to see how I look on a real GPU | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/Muon_run.txt) |
