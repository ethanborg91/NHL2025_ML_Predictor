import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Load raw data
    df_raw = pd.read_csv('all_teams.csv')
    print("Raw data shape:", df_raw.shape)
    print("Unique seasons:", sorted(df_raw['season'].unique()))  # Print to confirm 2020-2024 available

    # Process to game-level
    df_all = df_raw[df_raw['situation'] == 'all'].copy()

    # Filter to last 5 seasons (2020-2024 in dataset format; corresponds to 2020-21 to 2024-25)
    df_all_recent = df_all[df_all['season'] >= 2020]

    # Filter to home team rows
    df_home = df_all_recent[df_all_recent['home_or_away'] == 'HOME']

    # Rename/derive columns
    df_games = df_home[['gameDate', 'playerTeam', 'opposingTeam', 'goalsFor', 'goalsAgainst', 'season']].copy()
    df_games.rename(columns={
        'gameDate': 'date',
        'playerTeam': 'home_team',
        'opposingTeam': 'away_team',
        'goalsFor': 'home_goals',
        'goalsAgainst': 'away_goals'
    }, inplace=True)

    # Derive winner
    df_games['winner'] = df_games.apply(lambda row: 'home' if row['home_goals'] > row['away_goals'] else 'away', axis=1)

    # Save processed CSV (now limited to 2020-2025)
    df_games.to_csv('games.csv', index=False)
    print("Processed games.csv (2020-2025) saved. Shape:", df_games.shape)
    print(df_games.head())

    # Exploration
    print(df_games.describe())
    print(df_games['winner'].value_counts(normalize=True))

    # Plots
    sns.countplot(x='winner', data=df_games)
    plt.title('Home vs Away Wins (2020-2025)')
    plt.show()