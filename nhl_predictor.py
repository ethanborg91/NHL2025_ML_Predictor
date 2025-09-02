"""
Predict the outcome of the 2025-26 NHL season using a random forest classifier.
This script takes historical NHL regular season games in CSV format, builds summary statistics for each team (points, wins, otl, losses, goals for/against, and goal difference), and trains a RandomForestClassifier from scikit-learn to predict the final league position of each team in a subsequent season.

The model uses data up to the 2024-25 season for training. NHL has 32 fixed teams (no promotion/relegation).

Historical data can be downloaded from Hockey-Reference.com (e.g., https://www.hockey-reference.com/leagues/NHL_2024_games.html) and exported as CSV. Each file (nhl_2021-22.csv, etc.) lists every regular season game with columns: Date, Visitor, G (visitor goals), Home, G (home goals), [empty], Att., LOG, Notes (OT/SO if applicable).

Each CSV contains ~1,312 games (32 teams x 82 games / 2). The script parses goals and uses 'Notes' to determine if the loser gets 1 point (OT/SO).

For training, each team's stats from season n predict its position in season n+1.

Usage: python nhl_prediction.py
Requires: pandas, numpy, scikit-learn.
"""
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def parse_match_results(df: pd.DataFrame) -> pd.DataFrame:
    """Parse goals and add OT indicator.
    Hockey-Reference columns: Visitor, G (visitor), Home, G (home), Notes.
    Renames for consistency and adds 'is_ot' (1 if OT/SO, else 0).
    """
    df = df.copy()
    df.rename(columns={
        'Visitor': 'away_team',
        'G': 'away_goals',  # First G is visitor/away
        'Home': 'home_team',
        'G.1': 'home_goals',  # Second G is home
    }, inplace=True)
    df['away_goals'] = df['away_goals'].astype(int)
    df['home_goals'] = df['home_goals'].astype(int)
    df['is_ot'] = df['Notes'].apply(lambda x: 1 if x in ['OT', 'SO'] else 0)
    return df

def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarise a season into per-team statistics and final ranking.
    Computes points, wins, otl (OT losses), losses, goals_for/against, goal_diff.
    Sorts by points (desc), goal_diff (desc), goals_for (desc) for positions.
    """
    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "otl": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    })
    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        is_ot = row["is_ot"]
        # Update goals
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        # Determine outcome
        if hg > ag:
            # Home win
            teams[home]["points"] += 2
            teams[home]["wins"] += 1
            if is_ot:
                teams[away]["points"] += 1
                teams[away]["otl"] += 1
            else:
                teams[away]["losses"] += 1
        else:
            # Away win (NHL no ties)
            teams[away]["points"] += 2
            teams[away]["wins"] += 1
            if is_ot:
                teams[home]["points"] += 1
                teams[home]["otl"] += 1
            else:
                teams[home]["losses"] += 1
    # Build DataFrame
    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        data.append(
            {
                "team": team,
                "points": stats["points"],
                "wins": stats["wins"],
                "otl": stats["otl"],
                "losses": stats["losses"],
                "goals_for": stats["goals_for"],
                "goals_against": stats["goals_against"],
                "goal_diff": goal_diff,
            }
        )
    summary = pd.DataFrame(data)
    # Sort by points, goal diff, goals for
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary["position"] = summary.index + 1
    return summary

def prepare_training_data(season_files: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare training features and labels from seasons.
    Features from season n predict position in n+1.
    No defaults needed (NHL teams stable), but if a team is missing, skip or handle.
    """
    season_summaries: Dict[str, pd.DataFrame] = {}
    for file_path in season_files:
        raw = pd.read_csv(file_path)
        parsed = parse_match_results(raw)
        summary = summarise_season(parsed)
        season_summaries[file_path] = summary
    # Build training data
    feature_rows = []
    target_rows = []
    files_sorted = season_files
    for i in range(len(files_sorted) - 1):
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")
        curr_summary = season_summaries[files_sorted[i + 1]].copy().set_index("team")
        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:
                feats = prev_summary.loc[team][
                    ["points", "wins", "otl", "losses", "goals_for", "goals_against", "goal_diff"]
                ].to_dict()
            else:
                # Rare for NHL (e.g., expansion); skip or assign average (add logic if needed)
                continue
            feature_rows.append(feats)
            target_rows.append(row["position"])
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    # Latest features for prediction (assume same teams)
    last_summary = season_summaries[files_sorted[-1]].copy().set_index("team")
    latest_features_df = last_summary[
        ["points", "wins", "otl", "losses", "goals_for", "goals_against", "goal_diff"]
    ]
    return X_train, y_train, latest_features_df

def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Build pipeline with scaler and RandomForestClassifier."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model

def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predict positions using expected position from probabilities."""
    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]

def main():
    # Season files
    season_files = [
        os.path.join(os.path.dirname(__file__), "nhl_2021-22.csv"),
        os.path.join(os.path.dirname(__file__), "nhl_2022-23.csv"),
        os.path.join(os.path.dirname(__file__), "nhl_2023-24.csv"),
        os.path.join(os.path.dirname(__file__), "nhl_2024-25.csv"),
    ]
    X_train, y_train, latest_features = prepare_training_data(season_files)
    model = build_and_train_model(X_train, y_train)
    predictions = predict_league_table(model, latest_features)
    # Truncate to 32 teams
    predictions = predictions.iloc[:32].copy()
    print("Predicted NHL 2025-26 standings (1 = top):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(expected pos {row['expected_position']:.2f})"
        )

if __name__ == "__main__":
    main()