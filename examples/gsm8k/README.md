# GSM8K Training/Eval with vLLM

Minimal commands to train and evaluate GSM8K using a vLLM-hosted model and the provided scripts.

## Prerequisites

- At least 2 GPU(s). I used 2 A100s.
- Required packages: `datasets`, `math-verify`.

Install deps (once):
```bash
uv pip install datasets math-verify
```

## 1) Start vLLM server

In one terminal, launch vLLM on one GPU:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu_memory_utilization 0.8
```

Adjust `CUDA_VISIBLE_DEVICES`, `--gpu_memory_utilization`, or other vLLM flags for your hardware.

## 2) Train on GSM8K

In another terminal, run the training script.
Assuming you run it from the top-level of the repository, we need to add `environments/` into our Python path.
Example:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 uv run python examples/gsm8k/train_gsm8k.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --split train --limit 256 \
  --concurrency 32 --batch-size 2 \
  --train-steps 100 --group-size 8 \
  --eval-every 10 --eval-limit 100
```

Notes:
- `PYTHONPATH=.` ensures local imports resolve.
- Tune `--limit`, `--concurrency`, and `--train-steps` to fit your budget/VRAM.
- Training logs include loss, reward, avg_completion_length, and reducer stats (correct/parse-error rates, token totals).
- Checkpoints are written to `checkpoints_gsm8k/` by default.

## 3) Evaluate

Use the vLLM eval script to measure accuracy and reducer metrics:
```bash
PYTHONPATH=. uv run python examples/gsm8k/eval_gsm8k_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split test --limit 250 \
  --temperature 0.6 --max-tokens 512 \
  --concurrency 32 \
  --out gsm8k_eval.jsonl
```

Output includes accuracy plus reducer stats (correct_rate, parse_err_rate, avg_completion_length, total_completion_tokens). Results are also written to the JSONL path you set with `--out`.

## Tips

- Use `--start-server` in `eval_gsm8k_vllm.py` if you want the script to launch vLLM for you.
- Logs/rollouts are written to `gsm8k_train_rollouts.jsonl` (training) and your chosen `--out` path (eval).***
