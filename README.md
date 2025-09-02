NHL 2025-26 Season Predictor
This project uses machine learning to predict the final standings of the 2025-26 NHL regular season. It employs a Random Forest Classifier from scikit-learn, trained on historical team performance data from the 2018-19 to 2023-24 seasons, to forecast team positions based on aggregated stats like points, wins, overtime losses, goals for/against, and goal difference.
The model extrapolates from the most recent season (2024-25) to predict the upcoming one. As of September 2025, the 2025-26 season is just starting, so these are preseason predictions—fun for fantasy leagues, betting insights, or just hockey discussion!
Features

Data Processing: Parses NHL game results from CSV files, computes season summaries, and handles overtime/shootout points accurately.
Machine Learning: Trains a Random Forest Classifier to predict league positions (1-32).
Prediction Method: Uses expected positions derived from class probabilities for smooth ranking.
Output: A ranked list of all 32 NHL teams with expected positions.

Example output (based on 2024-25 data):
textPredicted NHL 2025-26 standings (1 = top):

1. Winnipeg Jets (expected pos 7.93)
2. Los Angeles Kings (expected pos 9.57)
3. Vegas Golden Knights (expected pos 10.00)
4. Tampa Bay Lightning (expected pos 10.32)
5. Minnesota Wild (expected pos 11.73)
   ... (truncated for brevity)
6. San Jose Sharks (expected pos 29.11)
   This prediction aligns reasonably with recent trends, e.g., strong Central Division teams like Winnipeg and Dallas staying competitive, while rebuilding squads like Chicago and San Jose lag behind.
   Data Sources
   Historical NHL game data is sourced from Hockey-Reference.com. Download regular season games for each year (e.g., 2024 games) and export as CSV via "Share & Export" > "Get table as CSV". Save files as nhl_YYYY-YY.csv (e.g., nhl_2024-25.csv).
   Included seasons: 2018-19 to 2024-25.
   Setup

Clone the Repository:
textgit clone <your-repo-url>
cd NHL2025_ML_Predictor

Create Virtual Environment (optional but recommended):
textpython -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install Dependencies:
textpip install -r requirements.txt
(Requires: pandas, numpy, scikit-learn)
Add Data Files: Place the CSV files (e.g., nhl_2018-19.csv to nhl_2024-25.csv) in the project root.

Usage
Run the script:
textpython nhl_predictor.py
This will train the model and print the predicted 2025-26 standings.
How It Works

Data Preparation: Summarizes each season's games into team stats.
Training: Uses stats from season n to predict positions in n+1.
Prediction: Applies the trained model to the latest season's stats.
Customization: Adjust hyperparameters in build_and_train_model (e.g., n_estimators, max_depth).

Limitations & Improvements

Assumes team stability (no expansion considered beyond current 32 teams).
Doesn't account for in-season factors like trades, injuries, or coaching changes.
Potential Enhancements:

Add more features (e.g., power play efficiency from MoneyPuck.com).
Evaluate accuracy by backtesting on past seasons.
Visualize predictions with matplotlib (e.g., bar charts of expected positions).

License
MIT License. Feel free to use and modify!
For questions or contributions, open an issue or PR. Built for learning and resume purposes—enjoy! 🏒
