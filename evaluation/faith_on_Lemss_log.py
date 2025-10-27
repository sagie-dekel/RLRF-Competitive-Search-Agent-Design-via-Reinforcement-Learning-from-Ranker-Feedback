import pandas as pd
import re
import json
from collections import defaultdict

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model_path = 'google/t5_11b_trueteacher_and_anli'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map={"": 0})
model.eval()

@torch.no_grad()
def get_prob_one_sided(premise, hypotheses, batch_size=16):
    inputs = [f"premise: {premise} hypothesis: {hyp}" for hyp in hypotheses]
    encodings = tokenizer(
        inputs,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    decoder_input_ids = torch.full(
        (len(hypotheses), 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=model.device
    )

    all_probs = []
    for i in range(0, len(hypotheses), batch_size):
        input_ids_batch = encodings['input_ids'][i:i+batch_size]
        attention_mask_batch = encodings['attention_mask'][i:i+batch_size]
        decoder_ids_batch = decoder_input_ids[i:i+batch_size]

        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            decoder_input_ids=decoder_ids_batch
        )
        logits = outputs.logits
        probs = torch.softmax(logits[:, 0, :], dim=-1)
        one_token_id = tokenizer('1').input_ids[0]
        batch_probs = probs[:, one_token_id].tolist()
        all_probs.extend(batch_probs)

    return all_probs

def calculate_metrics(prev_text, mod_text):
    sentences = [s.strip() for s in re.split(r'[.!?]', mod_text) if s]
    vals = get_prob_one_sided(prev_text, sentences)
    res = sum(v >= 0.5 for v in vals) / len(vals) if vals else 0

    ref_sentences = [s.strip() for s in re.split(r'[.!?]', prev_text) if s]
    ref_vals = get_prob_one_sided(prev_text, ref_sentences)
    ref_res = sum(v >= 0.5 for v in ref_vals) / len(ref_vals) if ref_vals else 0

    norm_res = min(res / ref_res if ref_res > 0 else 1.0, 1.0)
    return norm_res


save_path = ""

# Load CSV
competition_log_path = "competition_history.csv"
df = pd.read_csv(competition_log_path)

# calculate fithfullness by previous round or by first round
use_previous_round_as_reference = True

# Group by query_id and round
results = defaultdict(lambda: defaultdict(list))  # results[player][round] = list of scores

# For each query, use round 0 as reference
for query_id in df['query_id'].unique():
    q_df = df[df['query_id'] == query_id]
    round_0_df = q_df[q_df['round'] == 0]
    if round_0_df.empty:
        continue

    sorted_rounds = sorted(q_df['round'].unique())
    prev_docs_by_player = {}  # Keep track of previous round documents per player

    for rnd in sorted_rounds:
        round_df = q_df[q_df['round'] == rnd]

        for _, row in round_df.iterrows():
            player = row['player']
            mod_text = row['document']

            # Skip round 0 if using round 0 as reference
            if rnd == 0:
                if use_previous_round_as_reference:
                    prev_docs_by_player[player] = mod_text
                continue

            # Determine base_text
            if use_previous_round_as_reference:
                if player not in prev_docs_by_player:
                    continue  # No reference yet
                base_text = prev_docs_by_player[player]
            else:
                base_text = round_0_df.iloc[0]['document']
            print(base_text, rnd)
            score = calculate_metrics(base_text, mod_text)
            results[player][rnd].append(score)

            # Update previous round document
            prev_docs_by_player[player] = mod_text
print(results)
# Aggregate: average by query, then by round
final_output = {}
rounds_set = sorted(set(df['round'].unique()) - {0})

for player, round_scores in results.items():
    final_output[player] = {}
    for rnd in rounds_set:
        if rnd in round_scores:
            final_output[player][f"round_{rnd}"] = sum(round_scores[rnd]) / len(round_scores[rnd])
    # Average across rounds
    round_vals = [v for k, v in final_output[player].items() if k.startswith("round_")]
    if round_vals:
        final_output[player]["average_across_rounds"] = sum(round_vals) / len(round_vals)
print(final_output)
# Group: separate RLRF_agent and the rest
grouped_result = {
    "RLRF_agent": final_output.get("RLRF_env", {}),
    "others_avg": {}
}
# Average of others
others = [v for k, v in final_output.items() if k != "RLRF_agent"]
for rnd in rounds_set:
    key = f"round_{rnd}"
    vals = [p.get(key, 0) for p in others if key in p]
    grouped_result["others_avg"][key] = sum(vals) / len(vals) if vals else None

# Average across rounds
vals = [v for k, v in grouped_result["others_avg"].items() if k.startswith("round_") and v is not None]
grouped_result["others_avg"]["average_across_rounds"] = sum(vals) / len(vals) if vals else None
print(grouped_result)

# Save
with open(save_path, "w") as f:
    json.dump(grouped_result, f, indent=2)
