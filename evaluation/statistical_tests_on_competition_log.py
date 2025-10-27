import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

file_path = "competition_history.csv"
df = pd.read_csv(file_path, on_bad_lines='skip')

# Filter out round 0
df['round'] = pd.to_numeric(df['round'], errors='coerce')
df = df[df['round'] != 0].copy()

# Fix datatypes
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
df['game_id'] = df['game_id'].astype(str)
df['player'] = df['player'].str.strip()

# Keep only valid agents
valid_agents = df['player'].value_counts().head(10).index.tolist()
df = df[df['player'].isin(valid_agents)]

# Compute per-agent win rate
win_df = df.copy()
win_df['win'] = (win_df['rank'] == 1).astype(int)
win_rates = win_df.groupby('player')['win'].mean().sort_values(ascending=False)

# Identify second best player (excluding RLRF_agent)
second_best_player = win_rates.drop('RLRF_agent').idxmax()

# Pivot to get win (0/1) per round+game_id+player
pivot = win_df.pivot_table(index=['round', 'game_id'], columns='player', values='win')

# Print RLRF_agent win rate
rlrf_win_rate = pivot['RLRF_agent'].mean()
second_best_win_rate = pivot[second_best_player].mean()
print(f"RLRF_agent win rate: {rlrf_win_rate:.4f}")
print(f"Second best player win rate: {second_best_win_rate:.4f}")

# Perform paired t-test
t_stat, p_value = ttest_rel(pivot['RLRF_agent'], pivot[second_best_player], alternative='greater')

print(f"Total samples: {len(pivot)}")
print(f"Paired T-test result:")
print(f"  T-statistic: {t_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Second Best Player: {second_best_player}")

# Actual difference
observed_diff = (pivot['RLRF_agent'] - pivot[second_best_player]).mean()

# Permutation test
n_permutations = 10000
diffs = []

for _ in range(n_permutations):
    # Randomly swap the two columns for each row
    swap_mask = np.random.rand(len(pivot)) < 0.5
    shuffled_rlrf = pivot['RLRF_agent'].copy()
    shuffled_other = pivot[second_best_player].copy()

    # swap values where mask is True
    shuffled_rlrf[swap_mask], shuffled_other[swap_mask] = (
        shuffled_other[swap_mask], shuffled_rlrf[swap_mask]
    )

    # store permuted difference
    diff = (shuffled_rlrf - shuffled_other).mean()
    diffs.append(diff)

# Compute two-sided p-value
diffs = np.array(diffs)
p_value_perm = np.mean(diffs >= abs(observed_diff))

print(f"Permutation Test (n={n_permutations})")
print(f" Observed Mean Difference: {observed_diff:.4f}")
print(f" Permutation p-value: {p_value_perm:.4f}")

