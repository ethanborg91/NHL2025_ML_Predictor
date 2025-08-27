import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Load and filter raw data to 2020-2025
    df_raw = pd.read_csv('all_teams.csv')
    print("Raw data shape:", df_raw.shape)
    df_raw_recent = df_raw[df_raw['season'] >= 2020]  # Filter here
    print("Filtered raw shape (2020-2025):", df_raw_recent.shape)  # ~50k rows, smaller memory

    # Process to game-level
    df_all = df_raw_recent[df_raw_recent['situation'] == 'all'].copy()
    df_home = df_all[df_all['home_or_away'] == 'HOME']
    df_games = df_home[['gameDate', 'playerTeam', 'opposingTeam', 'goalsFor', 'goalsAgainst', 'season']].copy()
    df_games.rename(columns={
        'gameDate': 'date',
        'playerTeam': 'home_team',
        'opposingTeam': 'away_team',
        'goalsFor': 'home_goals',
        'goalsAgainst': 'away_goals'
    }, inplace=True)
    df_games['winner'] = df_games.apply(lambda row: 'home' if row['home_goals'] > row['away_goals'] else 'away', axis=1)

    # Save small processed CSV
    df_games.to_csv('games.csv', index=False)
    print("games.csv (2020-2025) saved. Shape:", df_games.shape)  # ~6k rows

    # Exploration
    print(df_games['winner'].value_counts(normalize=True))
    sns.countplot(x='winner', data=df_games)
    plt.title('Home vs Away Wins (2020-2025)')
    plt.show()