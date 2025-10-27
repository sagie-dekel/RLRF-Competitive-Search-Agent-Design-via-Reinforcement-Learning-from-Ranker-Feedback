import pandas as pd
from sklearn.model_selection import train_test_split
import ir_datasets
import os

# === CONFIG ===
queries_path = "/lv_local/home/sagie.dekel/ms_marco/2023_queries.tsv"
qrels_path = "/lv_local/home/sagie.dekel/ms_marco/2023.qrels.pass.withDupes.txt"
output_dir = "/lv_local/home/sagie.dekel/ms_marco/"

# === 1. Load queries ===
queries_df = pd.read_csv(queries_path, sep='\t', header=None, names=['qid_ms_marco', 'query'])

# === 2. Load qrels ===
qrels_df = pd.read_csv(qrels_path, delim_whitespace=True, header=None, names=['qid_ms_marco', 'Q0', 'doc_id', 'relevance'])

# === 3. Choose one relevant passage per query ===
relevant_docs = qrels_df[qrels_df['relevance'] == 3]
fallback_docs = qrels_df[qrels_df['relevance'] == 2]

merged_df = queries_df.copy()
merged_df['doc_id'] = merged_df['qid_ms_marco'].map(
    relevant_docs.groupby('qid_ms_marco')['doc_id'].first().to_dict()
)
merged_df['doc_id'] = merged_df.apply(
    lambda row: fallback_docs[fallback_docs['qid_ms_marco'] == row['qid_ms_marco']]['doc_id'].iloc[0]
    if pd.isna(row['doc_id']) and row['qid_ms_marco'] in fallback_docs['qid_ms_marco'].values else row['doc_id'],
    axis=1
)
merged_df = merged_df.dropna(subset=['doc_id'])
merged_df = merged_df.reset_index(drop=True)
merged_df['qid'] = merged_df.index

# === 4. Load passages using ir_datasets ===
print("Loading ir_datasets 'msmarco-passage-v2'...")
dataset = ir_datasets.load("msmarco-passage-v2")

# Make a fast doc_id → passage lookup dictionary
passage_ids_needed = set(merged_df['doc_id'])
doc_lookup = {}

print("Building lookup from dataset...")
for doc in dataset.docs_iter():
    if doc.doc_id in passage_ids_needed:
        doc_lookup[doc.doc_id] = doc.text
    if len(doc_lookup) == len(passage_ids_needed):
        break

# === 5. Map passages to merged_df ===
merged_df['document'] = merged_df['doc_id'].map(doc_lookup)
merged_df = merged_df.dropna(subset=['document'])

# === 6. Final formatting and save ===
final_df = merged_df[['qid', 'query', 'document']]
train_df, test_df = train_test_split(final_df, test_size=0.1, random_state=42)

train_df.to_csv(os.path.join(output_dir, "train_with_initial_doc.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_with_initial_doc.csv"), index=False)

qid_mapping = merged_df[['qid', 'qid_ms_marco', 'query']]
qid_mapping.to_csv(os.path.join(output_dir, "queries.csv"), index=False)

print(f"✅ Done! Train size: {len(train_df)}, Test size: {len(test_df)}")
