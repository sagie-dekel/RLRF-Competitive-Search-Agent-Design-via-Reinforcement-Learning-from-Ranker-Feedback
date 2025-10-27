import pandas as pd
import numpy as np
import os
import numpy as np
from scipy.stats import permutation_test

# Paths to data files
base_path = ""

# Assume .position format files
r4_path = os.path.join(base_path, "documents_round4.position")
r5_path = os.path.join(base_path, "generated_round5.position")

# Read data
df_r4 = pd.read_csv(r4_path, sep=" ", names=["doc_id", "rank_r4"])
df_r5 = pd.read_csv(r5_path, sep=" ", names=["doc_id", "rank_r5"])


# Extract normalized doc_id (remove ROUND-xx prefix)
df_r4['norm_doc_id'] = df_r4['doc_id'].apply(lambda x: "-".join(x.split('-')[1:])[3:])
df_r5['norm_doc_id'] = df_r5['doc_id'].apply(lambda x: "-".join(x.split('-')[1:])[3:])


# Merge rankings on normalized doc_id
df = pd.merge(df_r4, df_r5, on="norm_doc_id", suffixes=('_r4', '_r5'), how="right")
#df.loc[df['rank_r4'].isna(), 'rank_r4'] = df[df['rank_r4'].isna()]['rank_r5']
# Remove first rank players
df = df[~df['rank_r4'].isna()]


# Extract player id (pid) from normalized doc_id (last part after '-')
#df['pid'] = df['norm_doc_id'].apply(lambda x: x.split('-')[-1])

# Compute max possible promotion per doc
df.loc[df['rank_r4'] < df['rank_r5'], ['max_promotion']] = 4 - df['rank_r4']
df.loc[~(df['rank_r4'] < df['rank_r5']), ['max_promotion']] = df['rank_r4'] - 1

# Compute Scaled Promotion, avoid div by zero and Exclude first-ranked player in round 4
df = df[df['max_promotion'] != 0]
df['scaled_promotion'] = (df['rank_r4'] - df['rank_r5']) / df['max_promotion']

# Label documents from our agent
df['is_llm_ed'] = df['doc_id_r5'].str.startswith("llm_ed_")

# Prepare arrays for permutation test
our_agent_scores = df[df['is_llm_ed']]['scaled_promotion'].values
#our_agent_scores = np.concatenate([our_agent_scores, our_agent_scores, our_agent_scores])  # Add zero for comparison
other_players_scores = df[~df['is_llm_ed']]['scaled_promotion'].values
print(f"Number of documents from our agent: {len(our_agent_scores)}")
print(f"Number of documents from other players: {len(other_players_scores)}")

# Run one-sided unpaired permutation test
perm_test_result = permutation_test(
    (our_agent_scores, other_players_scores),
    statistic=lambda x, y: np.mean(x) - np.mean(y),
    alternative='greater',
    n_resamples=10000,
    random_state=1,
    permutation_type="independent"
)

# Summary results
summary = {
    "our_agent_mean_scaled_promotion": np.mean(our_agent_scores),
    "other_players_mean_scaled_promotion": np.mean(other_players_scores),
    "p_value": perm_test_result.pvalue,
}

# Print results
print("Summary:")
for k, v in summary.items():
    print(f"{k}: {v:.5f}")

