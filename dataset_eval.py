from datasets import load_dataset
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion
from minions.utils.energy_tracking import (
    PowerMonitor,
    cloud_inference_energy_estimate_w_model_attributes,
)

import pandas as pd
import tiktoken
import time
import os

# ======= CONFIGURATION =======
SPLIT = "Blog"  # Can also use "Wikipedia" or "Paper"
NUM_EXAMPLES = 5
MODEL_NAME = "llama3.2:1b"
REMOTE_MODEL = "gpt-4o-mini"
OUTPUT_CSV = f"minions_longeval_{SPLIT.lower()}_results.csv"
# =============================

# Load dataset
dataset = load_dataset("SiweiWu/LongEval", split=SPLIT)
subset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

print(dataset.column_names)

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Monitor
monitor = PowerMonitor(mode="auto", interval=1.0)

# Results
results = []

for i, example in enumerate(subset):
    prompt = example["title"]
    plan = example.get("bullet_points", "") or example.get("headlines", "")
    reference = example["text"]

    # Format prompt (direct or plan-based)
    if plan:
        task = f"Use the following plan to write a long article titled '{prompt}'.\n\nPlan:\n{plan}"
    else:
        task = f"Write a detailed article based on the title: {prompt}"

    prompt_tokens = len(enc.encode(task))
    print(f"\n[{i}] Running Minions on {SPLIT} example — {prompt_tokens} tokens")

    # Initialize clients and minion
    local_client = OllamaClient(
        model_name=MODEL_NAME,
        num_ctx=prompt_tokens + 1500,
        max_tokens=1500,
    )
    remote_client = OpenAIClient(model_name=REMOTE_MODEL)
    minion = Minion(local_client, remote_client)

    # Run with power monitoring
    monitor.start()
    output = minion(task=task, context=[], max_rounds=2)
    monitor.stop()

    # Extract token usage
    local_tokens = output["local_usage"].prompt_tokens + output["local_usage"].completion_tokens
    remote_tokens = output["remote_usage"].prompt_tokens + output["remote_usage"].completion_tokens

    # Minions energy usage
    local_energy_joules = float(monitor.get_final_estimates()["Measured Energy"].split(" ")[0])
    remote_energy = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=output["remote_usage"].prompt_tokens,
        output_tokens=output["remote_usage"].completion_tokens,
    )
    total_energy = local_energy_joules + remote_energy["total_energy_joules"]

    # Baseline energy estimate
    baseline_energy = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=output["local_usage"].prompt_tokens,
        output_tokens= output["local_usage"].completion_tokens,
    )
    baseline_total_joules = baseline_energy["total_energy_joules"]
    minions_vs_baseline_delta = total_energy - baseline_total_joules

    # Store results
    results.append({
        "example_id": i,
        "split": SPLIT,
        "prompt_text": task,
        "prompt_tokens": prompt_tokens,
        "local_tokens": local_tokens,
        "remote_tokens": remote_tokens,
        "measured_local_energy_j": local_energy_joules,
        "remote_estimated_energy_j": remote_energy["total_energy_joules"],
        "total_energy_j": total_energy,
        "baseline_remote_energy_j": baseline_total_joules,
        "minions_vs_baseline_delta_j": minions_vs_baseline_delta,
        "minions_output": output["final_answer"],
        "reference": reference
    })

# Save to CSV
df = pd.DataFrame(results)
csv_path = os.path.join(os.getcwd(), OUTPUT_CSV)
df.to_csv(csv_path, index=False)
print(f"\Results saved to: {csv_path}")
