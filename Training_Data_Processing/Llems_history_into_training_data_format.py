import pandas as pd

# Load the uploaded CSV file
file_path = "competition_history.csv"
df = pd.read_csv(file_path)

# Rounds to keep from competition log
keep_rounds = [3]

# Filter out not relevant rounds
df_filtered = df[df['round'] not in keep_rounds].copy()
df_filtered['combined_prompt'] = "<|start_header_id|>system<|end_header_id|> " + df_filtered['system_prompt'].fillna('') + '\n' + "<|start_header_id|>user<|end_header_id|>" + df_filtered['user_prompt'].fillna('')

# Ensure rank column is numeric
df_filtered['rank'] = pd.to_numeric(df_filtered['rank'], errors='coerce')

# Initialize list to store new rows
output_rows = []

# Group by game_id and round
for (game_id, rnd), group in df_filtered.groupby(['game_id', 'round']):
    if group.empty:
        continue

    # Get highest (rank == 1) and lowest (max rank) ranked players
    highest = group.loc[group['rank'].idxmin()]
    lowest = group.loc[group['rank'].idxmax()]

    # Row where prompt is highest ranked player's document
    output_rows.append({
        "game_id": game_id,
        "round": rnd,
        "prompt": highest['combined_prompt'],
        "chosen": highest['document'],
        "rejected": lowest['document']
    })

    # For pairwise:
    """
    # Row where prompt is lowest ranked player's document
    output_rows.append({
        "game_id": game_id,
        "round": rnd,
        "prompt": lowest['combined_prompt'],
        "chosen": highest['document'],
        "rejected": lowest['document']
    })
    """
# Convert to DataFrame
output_df_all_games = pd.DataFrame(output_rows)

# Save to CSV
output_csv_path = "C:/Users/sagie/Downloads/mistral_training_data_listwise_prompt_0.5_temp.csv"
output_df_all_games.to_csv(output_csv_path, index=False)
