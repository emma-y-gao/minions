from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient

from minions.minion import Minion

from minions.utils.energy_tracking import (
    PowerMonitor,
    cloud_inference_energy_estimate_w_model_attributes,
)

import tiktoken
import pandas as pd
import os

# Setup power monitor
monitor = PowerMonitor(mode="auto", interval=1.0)

path = "minions/examples/health/sample.txt"

# Read in the .txt file
with open(path, "r") as f:
    context = f.read()

enc = tiktoken.get_encoding("cl100k_base")
base_context_tokens = len(enc.encode(context))
print(f"Base context size in tokens: {base_context_tokens}")

# Create a dataframe to store results
results = []

# Define the task
task = "Answer the following question.\n\nOn 01/09/2017, Mr. Williams received a CT scan of her whole body. What kind of contrast agent was used?\n\nOptions:\na) Omnipaque 320\nb) Imeron 400\nc) Optiray 320\nd)Omnipaque 240\ne)Ultravist 370"

# Run experiment for different context sizes
for i in range(5):

    print(f"\n--- Running experiment with context multiplier: {i+1} ---")

    # Repeat context i+1 times
    extended_context = context * (i + 1)
    context_tokens = len(enc.encode(extended_context))
    print(f"Context size in tokens: {context_tokens}")

    # Initialize clients for each run
    local_client = OllamaClient(
        model_name="llama3.2:1b",
        num_ctx=context_tokens + 1000,
        max_tokens=context_tokens + 1000 + 500,
    )

    remote_client = OpenAIClient(
        model_name="gpt-4o-mini",
    )

    minion = Minion(local_client, remote_client)

    # Execute the minion protocol
    monitor.start()
    output = minion(task=task, context=[extended_context], max_rounds=2)
    monitor.stop()

    # Get energy metrics
    local_energy = monitor.get_final_estimates()

    # Calculate cloud energy estimate
    local_tokens = (
        output["local_usage"].completion_tokens + output["local_usage"].prompt_tokens
    )

    remote_tokens = (
        output["remote_usage"].completion_tokens + output["remote_usage"].prompt_tokens
    )

    # print(output.keys())
    # remote_time = output["timing"]["remote_call_time"]

    minions_remote_energy = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=remote_tokens,
        output_tokens=remote_tokens,
    )
    print(f"Minions Remote Energy: {minions_remote_energy}")

    # ===============================
    import time

    input = f"Context: {extended_context}\n\n{task}\n\nAnswer:"
    start_time = time.time()
    baseline_output, usage = remote_client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ]
    )
    end_time = time.time()
    baseline_time = end_time - start_time
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    print(f"Baseline Time: {baseline_time} seconds")

    # ===============================
    cloud_only_estimate = cloud_inference_energy_estimate_w_model_attributes(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # ===============================

    # Print results for this run
    print(f"\nMeasured Energy and Power Metrics --- (Minions) Local Client")
    for key, value in local_energy.items():
        print(f"{key}: {value}")

    print(f"Local tokens: {local_tokens}")
    print(
        f"(Baseline) Remote Only Energy Estimate: {cloud_only_estimate['total_energy_joules']} J"
    )

    # final_estimates["Measured Energy"] =  "1590.81 J"
    # convert this to a float
    measured_energy = float(local_energy["Measured Energy"].split(" ")[0])

    print(f"Measured Energy: {measured_energy} J")

    minions_total_energy = (
        minions_remote_energy["total_energy_joules"] + measured_energy
    )

    print(f"(Minions) Total Energy Estimate: {minions_total_energy} J")

    # Store results
    result_row = {
        "context_multiplier": i + 1,
        "total_context_tokens": context_tokens,
        "cloud_energy_joules": cloud_only_estimate["total_energy_joules"],
        "cloud_prefill_energy_joules": cloud_only_estimate["prefill_energy_joules"],
        "cloud_decoding_energy_joules": cloud_only_estimate["decoding_energy_joules"],
        "minions_remote_energy_joules": minions_remote_energy["total_energy_joules"],
        "minions_remote_prefill_energy_joules": minions_remote_energy[
            "prefill_energy_joules"
        ],
        "minions_remote_decoding_energy_joules": minions_remote_energy[
            "decoding_energy_joules"
        ],
        "minions_total_energy_joules": minions_total_energy,
        
        # Tokens
        "minions_local_tokens": local_tokens,
        "minions_remote_tokens": remote_tokens,
        "minions_local_prompt_tokens": output["local_usage"].prompt_tokens,
        "minions_local_completion_tokens": output["local_usage"].completion_tokens,
        "minions_remote_prompt_tokens": output["remote_usage"].prompt_tokens,
        "minions_remote_completion_tokens": output["remote_usage"].completion_tokens,
    }

    # Add all local energy metrics with local_ prefix (in case there are any additional metrics not explicitly listed above)
    for key, value in local_energy.items():
        key = key.lower().replace(" ", "_")
        prefixed_key = f"local_{key}"
        if prefixed_key not in result_row:
            result_row[prefixed_key] = value

    results.append(result_row)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "energy_ablation_results_3b.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

# Display summary
print("\nSummary of results:")
print(
    results_df[
        [
            "context_multiplier",
            "total_context_tokens",
            "local_measured_energy",
            "cloud_energy_joules",
        ]
    ]
)
