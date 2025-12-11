"""
Eval Qwen2.5-7B-Instruct (or any vLLM-served model) on GSM8K.

Assumes a running vLLM OpenAI server. By default we run the GSM8K
test split and report accuracy. Uses math-verify for grading.

Example:
    uv run python examples/gsm8k/eval_gsm8k_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --host 127.0.0.1 --port 8000 \
        --limit 200

Requires: uv pip install datasets math-verify
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import List, Sequence, Dict

import requests

from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import ParseResult, cot_prefix_parser
from ludic.types import SamplingArgs
from environments.gsm8k import GSM8KEnv
from ludic.training.stats import Reducer, apply_reducers_to_records


def load_gsm8k(split: str, limit: int | None) -> List[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This example requires the 'datasets' package. "
            "Install with: uv pip install datasets"
        ) from e

    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[dict] = []
    for idx, row in enumerate(ds):
        samples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "id": row.get("id", idx),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"No GSM8K samples loaded for split={split}")
    return samples


def optional_cot_parser(raw: str) -> ParseResult:
    """
    Try to strip a <think>...</think> prefix; otherwise keep raw text.
    """
    result = cot_prefix_parser(raw)
    if result.action is None:
        return ParseResult(action=raw.strip(), reward=0.0, obs=None)
    return result


def start_vllm_server(model: str, host: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cmd = [
        sys.executable,
        "-m",
        "ludic.inference.vllm_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu_memory_utilization",
        "0.7",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
    ]

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for_health(host: str, port: int, proc: subprocess.Popen, timeout_s: float = 180.0) -> None:
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e

        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        time.sleep(2.0)

    proc.terminate()
    stdout, stderr = proc.communicate(timeout=10)
    raise RuntimeError(
        f"vLLM server failed to become healthy at {health_url}\n"
        f"Last error: {last_err}\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )


async def eval_dataset(
    *,
    dataset: Sequence[dict],
    model: str,
    host: str,
    port: int,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    timeout_s: float | None,
    concurrency: int = 1,
) -> List[dict]:
    reducers: Dict[str, Reducer] = {
        "correct_rate": Reducer(kind="count_true", source="correct", normalize_by="samples"),
        "parse_err_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples"),
        "avg_completion_length": Reducer(kind="mean", source="completion_length"),
        "total_completion_tokens": Reducer(kind="sum", source="completion_length"),
    }

    client = VLLMChatClient(
        host=host,
        port=port,
        enable_weight_updates=False,
    )

    sargs: SamplingArgs = {"temperature": temperature, "max_tokens": max_tokens}

    total = 0
    correct = 0
    parse_errors = 0
    records: List[dict] = []

    samples = list(dataset)
    idx = 0
    while idx < len(samples):
        batch_samples = samples[idx : idx + concurrency]
        batch_size = len(batch_samples)
        tasks = []
        for sample in batch_samples:
            # Each episode uses a single-sample env to avoid reuse
            e = GSM8KEnv(sample=sample, system_prompt=system_prompt)
            tasks.append(
                SingleAgentSyncProtocol(
                    agent=Agent(
                        client=client,
                        model=model,
                        ctx=FullDialog(),
                        parser=optional_cot_parser,
                    )
                ).run(
                    env=e,
                    max_steps=1,
                    sampling_args=sargs,
                    timeout_s=timeout_s,
                )
            )

        batch_rollouts = await asyncio.gather(*tasks)
        idx += batch_size
        for rollouts in batch_rollouts:
            step = rollouts[0].steps[-1]
            info = step.info

            total += 1
            if info.get("correct"):
                correct += 1
            if info.get("parse_error") or step.truncated:
                parse_errors += 1

            completion_ids = info.get("completion_token_ids") or []

            records.append(
                {
                    "question_id": info.get("question_id"),
                    "sample_index": info.get("sample_index"),
                    "question": rollouts[0].steps[0].prev_obs,
                    "raw_action": step.action,
                    "parsed_answer": info.get("parsed_answer"),
                    "target_answer": info.get("target_answer"),
                    "correct": info.get("correct"),
                    "parse_error": info.get("parse_error"),
                    "reward": step.reward,
                    "truncated": step.truncated,
                    "terminated": step.terminated,
                    "completion_length": len(completion_ids) if isinstance(completion_ids, list) else 0,
                }
            )

        acc = 100 * correct / total
        print(f"[{total}/{len(dataset)}] accuracy={acc:.2f}% parse_errors={parse_errors}")

    accuracy = 100 * correct / total
    print("---- GSM8K Evaluation ----")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Parse errors  : {parse_errors}")
    reducer_stats = apply_reducers_to_records(records, reducers)
    if reducer_stats:
        print("Reducer stats :")
        for k, v in reducer_stats.items():
            print(f"  {k}: {v}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a vLLM server (optional) and evaluate a model on GSM8K.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="If set, launch a local vLLM server for the chosen model before eval.",
    )
    parser.add_argument("--split", type=str, default="test", help="GSM8K split to use.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max samples.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a careful math tutor. Think in <think></think> and put your final numeric answer after '####'.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=None, help="Per-call timeout.")
    parser.add_argument(
        "--out",
        type=str,
        default="gsm8k_rollouts.jsonl",
        help="Path to write rollout results as JSONL.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of parallel episodes to run.",
    )

    args = parser.parse_args()

    dataset = load_gsm8k(args.split, args.limit)
    print(f"Loaded {len(dataset)} GSM8K samples from split '{args.split}'")
    print(f"Evaluating model '{args.model}' via vLLM at {args.host}:{args.port}")

    proc = None
    if args.start_server:
        print("Starting local vLLM server...")
        proc = start_vllm_server(args.model, args.host, args.port)
        try:
            wait_for_health(args.host, args.port, proc)
            print("vLLM server is healthy.")
        except Exception:
            proc.kill()
            raise

    try:
        records = asyncio.run(
            eval_dataset(
                dataset=dataset,
                model=args.model,
                host=args.host,
                port=args.port,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                concurrency=args.concurrency,
            )
        )
        if records:
            import json
            out_path = args.out
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Wrote {len(records)} records to {out_path}")
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
