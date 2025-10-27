import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test

import torch
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer
import re
from work.offline_evaluation_experiment import QueryDocLoader

## init TT model
model_path = 'google/t5_11b_trueteacher_and_anli'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                                   use_safetensors=True)
model.eval()


## calculate TT(premise, hypothesis)
@torch.no_grad()
def get_prob_one_sided(premise, hypothesis):
    global tokenizer
    input_ids = tokenizer(
        f'premise: {premise} hypothesis: {hypothesis}',
        return_tensors='pt',
        truncation=True,
        max_length=512).input_ids
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits
    probs = torch.softmax(logits[0], dim=-1)
    one_token_id = tokenizer('1').input_ids[0]
    entailment_prob = probs[0, one_token_id].item()
    return entailment_prob


## calculate OrigFaith RF(d_mod, d_curr) / RF(d_curr, d_curr)
def calculate_metrics(prev_text, sentences, cache):  # prev_text = d_curr
    cache = {}
    ## calculate RF(d_mod, d_curr)
    vals = []
    for sentence in sentences:
        cache[sentence] = get_prob_one_sided(prev_text, sentence)
        vals.append(cache[sentence])
    bool_vals = [v >= 0.5 for v in vals]
    res = sum(bool_vals) / len(bool_vals)

    ## calculate RF(d_curr, d_curr)
    ref_vals = []
    ref_sentences = [sentence.strip() for sentence in re.split(r'[.!?]', prev_text) if sentence]
    for ref_sentence in ref_sentences:
        cache[ref_sentence] = get_prob_one_sided(prev_text, ref_sentence)
        ref_vals.append(cache[ref_sentence])
    ref_bool_vals = [v >= 0.5 for v in ref_vals]
    ref_res = sum(ref_bool_vals) / len(ref_bool_vals)

    norm_res = min(res / ref_res if ref_res != 0 else 1.0, 1.0)
    return res, ref_res, norm_res


## main
## Load files
base_path = ""

queries_path = os.path.join(base_path, "queries.txt")
docs_trectext_path = os.path.join(base_path, "documents.trectext")
docs_round4_path = os.path.join(base_path, "documents_round4_full.position")
docs_round5_path = os.path.join(base_path, "generated_round5.position")
docs_generated_path = os.path.join(base_path, "generated_documents5.trectext")
base_round = 5

## Load queries
queries = {}
with open(queries_path, 'r') as f:
    for line in f:
        qid, qtext = line.strip().split(' ', 1)
        queries[qid] = qtext

## Load documents_round4.position
round4_ranks = {}
with open(docs_round4_path, 'r') as f:
    for line in f:
        doc_id, rank = line.strip().split()
        rank = int(rank)
        qid = doc_id.split("-")[2].split("_")[0]
        if qid not in round4_ranks:
            round4_ranks[qid] = []
        round4_ranks[qid].append((rank, doc_id))

## Load generated_round5.position
round5_ranks = []
with open(docs_round5_path, 'r') as f:
    for line in f:
        doc_id, rank = line.strip().split()
        round5_ranks.append((int(rank), doc_id))

## Load generated_documents5.trectext
generated_docs = {}
with open(docs_generated_path, 'r') as f:
    current_doc_id = None
    text_lines = []
    for line in f:
        line = line.strip()
        if line.startswith("<DOCNO>"):
            current_doc_id = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
        elif line.startswith("</DOC>") and current_doc_id:
            generated_docs[current_doc_id] = "\n".join(text_lines).strip()
            current_doc_id = None
            text_lines = []
        elif current_doc_id:
            if not line.startswith("<TEXT>") and not line.startswith("</TEXT>"):
                text_lines.append(line)

loader = QueryDocLoader(
    query_path=queries_path,
    doc_path=docs_trectext_path,
    round4_path=docs_round4_path,
    history_path=docs_round4_path,  # you can pass round4 here just to satisfy the init
    round5_path=docs_round5_path
)
our_agent_scores = []
other_players_scores = []
our_agent_results = []
other_agent_results = []

## process your agent (LLM doc)
print("Processing LLM-ed documents from round 5...", flush=True)
i = 0
for rank, doc_id in round5_ranks:
    if doc_id.startswith("llm_ed_"):
        print(f"{i}", flush=True)
        i += 1
        qid = doc_id.split("-")[2].split("_")[0]

        ## find current doc from round 4 that this LLM-ed doc modified
        ## usually you use ROUND-04-<qid>_<pid>_... from round4
        curr_doc = None
        for r, d4_doc_id in round4_ranks[qid]:
            if f"ROUND-0{base_round - 1}" in d4_doc_id and doc_id.endswith(d4_doc_id[8:]):
                curr_doc = d4_doc_id
                break

        if curr_doc is None:
            print(f"Warning: cannot find matching d_curr for {doc_id}")
            continue

        # print(f"doc_id: {doc_id}", flush=True)
        # print(f"curr_doc: {curr_doc}", flush=True)

        d_curr_text = loader.docs[curr_doc]  ## or load manually from documents.trectext
        d_mod_text = generated_docs[doc_id]

        # print(f"d_mod_text: {d_mod_text}", flush=True)
        # print(f"d_curr_text: {d_curr_text}", flush=True)

        # Split sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', d_mod_text) if s]

        # print(f"sentences: {sentences}", flush=True)

        # Calculate
        cache = {}
        res, ref_res, norm_res = calculate_metrics(d_curr_text, sentences, cache)
        our_agent_scores.append(norm_res)
        our_agent_results.append(f"{doc_id}\t{res:.4f}\t{ref_res:.4f}\t{norm_res:.4f}")
        print(f"LLM doc {doc_id}: RF={res:.4f}, RF_ref={ref_res:.4f}, OrigFaith={norm_res:.4f}", flush=True)

# process other agents (round 4 rank != 1), using generated_documents5.trectext
# no duplicates
print("Processing other agents' documents from round 5...")
used_generated_ids = set()

# index generated_round5.position by query
round5_ranks_dict = {}
for rank, doc_id in round5_ranks:
    qid = doc_id.split("-")[2].split("_")[0]
    if qid not in round5_ranks_dict:
        round5_ranks_dict[qid] = []
    round5_ranks_dict[qid].append((rank, doc_id))
i = 0
for qid, rank_doc_list in round4_ranks.items():
    print(f"{i}")
    i += 1
    for rank, doc_id in rank_doc_list:
        """
        if rank == 1:
            continue  # skip first rank (best agent)
        """

        # expected generated DOC_ID: ROUND-05-xxx matching ROUND-04-xxx
        expected_suffix = doc_id[8:]  # everything after "ROUND-04-"

        matching_gen_doc_id = None

        if qid in round5_ranks_dict:
            for r5_rank, r5_doc_id in round5_ranks_dict[qid]:
                if r5_doc_id.startswith(f"ROUND-0{base_round}-") and r5_doc_id.endswith(
                        expected_suffix) and r5_doc_id not in used_generated_ids:
                    matching_gen_doc_id = r5_doc_id
                    used_generated_ids.add(r5_doc_id)
                    break

        if matching_gen_doc_id is None:
            print(f"Warning: cannot find generated doc for other agent {doc_id}")
            continue

        # print(f"doc_id: {doc_id}")
        # print(f"matching_gen_doc_id: {matching_gen_doc_id}")

        d_curr_text = loader.docs[doc_id]  # original round 4 doc
        d_mod_text = loader.docs[matching_gen_doc_id]  # generated round 5 doc

        # print(f"d_mod_text: {d_mod_text}")
        # print(f"d_curr_text: {d_curr_text}")

        sentences = [s.strip() for s in re.split(r'[.!?]', d_mod_text) if s]

        cache = {}
        res, ref_res, norm_res = calculate_metrics(d_curr_text, sentences, cache)
        other_players_scores.append(norm_res)
        other_agent_results.append(f"{matching_gen_doc_id}\t{res:.4f}\t{ref_res:.4f}\t{norm_res:.4f}")
        print(
            f"Other agent {matching_gen_doc_id} vs {doc_id}: RF={res:.4f}, RF_ref={ref_res:.4f}, OrigFaith={norm_res:.4f}")

print(f"our_agent_scores len: {len(our_agent_scores)}")
print(f"other_players_scores len: {len(other_players_scores)}")

perm_test_result = permutation_test(
    (other_players_scores, our_agent_scores),
    statistic=lambda x, y: np.mean(x) - np.mean(y),
    alternative='greater',
    n_resamples=10000,
    random_state=1,
    permutation_type="independent"
)

# Paths
llm_results_path = os.path.join(base_path, "faithfulness_llm.tsv")
other_results_path = os.path.join(base_path, "faithfulness_other_agents.tsv")

# Save LLM agent results
with open(llm_results_path, 'w') as f:
    f.write("DOC_ID\tRF\tRF_ref\tOrigFaith\n")
    f.write("\n".join(our_agent_results))

# Save other agents results
with open(other_results_path, 'w') as f:
    f.write("DOC_ID\tRF\tRF_ref\tOrigFaith\n")
    f.write("\n".join(other_agent_results))

# print(f"\nPermutation test result: p-value = {perm_test_result.pvalue:.4f}")
print(f"our_agent_faith: {np.mean(our_agent_scores)}")
print(f"other_players_faith: {np.mean(other_players_scores)}")

# Prepare result dictionary
result_summary = {
    "our_agent": {
        "mean_origfaith": float(np.mean(our_agent_scores)),
        "num_documents": len(our_agent_scores)
    },
    "other_agents": {
        "mean_origfaith": float(np.mean(other_players_scores)),
        "num_documents": len(other_players_scores)
    },
    "permutation_test": {
        "statistic_mean_diff": float(np.mean(our_agent_scores) - np.mean(other_players_scores)),
        "p_value": perm_test_result.pvalue,
        "n_resamples": 10000,
        "alternative": "greater",
        "test_type": "independent permutation test"
    }
}

# Save to JSON
results_path = os.path.join(base_path, "faithfulness_results_summary.json")
with open(results_path, "w") as f:
    json.dump(result_summary, f, indent=2)

