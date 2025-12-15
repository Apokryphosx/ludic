# Tic-Tac-Toe GRPO + LoRA

Small end-to-end RL run that fine-tunes a model with LoRA on a toy Tic-Tac-Toe env using grouped advantages (GRPO-style). Assumes 2 GPUs: GPU0 for vLLM inference, GPU1 for training.

## 1) Start vLLM (GPU0)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-7B-Instruct
```

## 2) Train (GPU1)
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/tic_tac_toe/train_tic_tac_toe.py \
  --model Qwen/Qwen2.5-7B-Instruct
```
- Uses `GRPORequestStrategy` + `GroupNormalizedReturn`, LoRA on the HF model, and a strict `<think>...</think><move>...</move>` parser.
- Tweak `--group-size`, `--concurrency`, `--train-steps`, `--train-temperature`, `--batch-size`, and `--max-steps-per-episode` as needed.

## 3) Evaluate (optional)
```bash
PYTHONPATH=. uv run python examples/tic_tac_toe/eval_tic_tac_toe_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --episodes 200 --temperature 0.6 --max-tokens 250
```
- Reports win/loss/draw/illegal/parse-error rates; writes `tictactoe_eval.jsonl` by default.
