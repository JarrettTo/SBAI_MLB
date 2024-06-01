import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import scrape, features
import dataframe_utilities as util
import pickle
import xgboost as xgb
import json
import os
def process_data():
    root_dir = os.getenv('ROOT_DIR', '')
    os.chdir(root_dir)
    print("ROOT_DIR:", root_dir)
    file_path = os.path.join(root_dir, '../data/game_summaries.csv')
    last_day = pd.to_datetime(pd.read_csv(file_path)['date']).max()
    get_day = last_day + pd.Timedelta(days=1)
    date_obj = datetime.date.today()
    while get_day.date() < date_obj:
        links = scrape.get_game_links(get_day)
        for l in links:
            scrape.process_link(l)
        get_day += + pd.Timedelta(days=1)
    test_df = scrape.get_today_games()
    test_df['is_night_game'] = True
    test_df['time_numeric'] = pd.to_numeric(test_df['time'].str[:1], errors='coerce')
    test_df.loc[(test_df['time_numeric'] < 5) & (test_df['time_numeric'].notna()), 'is_night_game'] = False
    test_df.loc[test_df['time_numeric'].isna(), 'is_night_game'] = True
    test_df['is_night_game'][test_df['time'].str[1:2].isin(['0','1'])] = True #for 10,11 PM games
    test_df.drop(columns='time', inplace=True)
    test_df['is_test'] = True
    test_df['home_team_win']=np.nan
    test_df['game_id'] = test_df.home_team_abbr + test_df.date.astype('str').str.replace('-','') + '0'
    df = features.get_game_df()
    df['is_test'] = False
    df = pd.concat([df,test_df])
    df = df.sort_values(by='date').reset_index(drop=True)
    df = features.add_trueskill_ratings(df)
    df = features.add_rest_durations(df)
    df.drop(columns=["is_test", "time_numeric"], inplace=True)
    date = pd.to_datetime(df['date'])
    df['season'] = date.dt.year
    df['month']=date.dt.month
    df['week_num'] = date.dt.isocalendar().week
    df['dow']=date.dt.weekday.astype('int')
    df['dh_game_no'] = pd.to_numeric(df['game_id'].str[-1:],errors='coerce')
    df['date'] = (pd.to_datetime(df['date']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') #epoch time
    games = features.get_games()
    batting = features.get_batting()
    pitching = features.get_pitching()
    pitchers = features.get_pitchers()
    b_stats = ['batting_avg','leverage_index_avg', 'onbase_perc', 'onbase_plus_slugging']
    df = features.add_10RA_rolling(batting, df, b_stats, True,'batting')
    pitching['SO_batters_faced'] = pitching['so'] / pitching['batters_faced']
    pitching['H_batters_faced'] = pitching['h'] / pitching['batters_faced']
    pitching['BB_batters_faced'] = pitching['bb'] / pitching['batters_faced']

    # create rolling stat
    b_stats = ['earned_run_avg','SO_batters_faced','H_batters_faced','BB_batters_faced']
    df = features.add_10RA_rolling(pitching, df, b_stats, True, 'team_pitching')
    pitchers['earned_run_avg'] = pd.to_numeric(pitchers['earned_run_avg'], errors='coerce')
    pitchers['SO_batters_faced'] = pitchers['so'] / pitchers['batters_faced']
    pitchers['H_batters_faced'] = pitchers['h'] / pitchers['batters_faced']
    pitchers['BB_batters_faced'] = pitchers['bb'] / pitchers['batters_faced']

    # create rolling stat
    b_stats = ['earned_run_avg','SO_batters_faced','H_batters_faced','BB_batters_faced']
    df = features.add_10RA_rolling(pitchers, df, b_stats, False, 'pitcher')
    df = features.game_stats(games,df)
    batting_stats = ['a', 'ab', 'bb', 'h', 'pa', 'po', 'r', 'rbi', 'so', 'batting_avg',
             'leverage_index_avg', 'onbase_perc', 'onbase_plus_slugging', 'pitches', 
             're24_bat', 'slugging_perc', 'strikes_total', 'wpa_bat', 'wpa_bat_neg', 
             'wpa_bat_pos']
    df = features.add_season_rolling(batting, df, batting_stats, True,'batting')
    pitching_stats = ['bb', 'er', 'h', 'hr', 'ip', 'r', 'so', 'batters_faced',
               'earned_run_avg', 'game_score', 'inherited_runners',
               'inherited_score', 'inplay_fb_total', 'inplay_gb_total', 'inplay_ld',
               'inplay_unk', 'leverage_index_avg', 'pitches', 're24_def',
               'strikes_contact', 'strikes_looking', 'strikes_swinging',
               'strikes_total', 'wpa_def','SO_batters_faced','H_batters_faced',
                'BB_batters_faced']
    df = features.add_season_rolling(pitching, df, pitching_stats, True,'team_pitching')
    if 'away_pitcher' not in df.columns:
        df['away_pitcher'] = 'Unknown'
    else:
        df['away_pitcher'].fillna('Unknown', inplace=True)
    df = features.add_season_rolling(pitchers, df, pitching_stats, False,'pitcher')
    df = util.fix_na(df, False)
    filtered_df = df[df['game_id'].isin(test_df['game_id'])]
    X_test = filtered_df.drop(columns=[
        'home_team_win', 'home_team_abbr', 'away_team_abbr', 'game_id',
        'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season'
    ])
    return X_test, test_df

def get_win_preds(X_test):
    model = xgb.XGBClassifier()
    model.load_model("../models/team_wins_model.json")
    model_features = model.get_booster().feature_names
    df_for_prediction = X_test[model_features]
    predictions = model.predict(df_for_prediction)
    proba = model.predict_proba(df_for_prediction)
    return predictions, proba

def get_runs_preds(X_test):
    model = xgb.XGBRegressor()
    model.load_model("../models/total_score_model.json")
    model_features = model.get_booster().feature_names
    df_for_prediction = X_test[model_features]
    score_predictions = model.predict(df_for_prediction)
    return score_predictions


def assemble_json(predictions, proba, score_predictions, test_df):
    selected_abbr = np.where(np.array(predictions) == 1, test_df['home_team_abbr'], test_df['away_team_abbr'])
    json_predictions = []
    for index, row in test_df.iterrows():
        prediction = {
            "id": test_df['game_id'],
            "home_team": row['home_team_abbr'],
            "away_team": row['away_team_abbr'],
            "ml_pred": selected_abbr[index],
            "ml_conf": str(max(proba[index])),
            "ou_pred": str(score_predictions[index]), 
            "ou_conf": "0"   # Placeholder values as specified
        }
        json_predictions.append(prediction)

    # Convert the list of dictionaries to JSON
    json_output = json.dumps(json_predictions, indent=2)
    return json_output

def main():
    print("Hello, this is the main function!")
    X_test, test_df = process_data()
    predictions, proba = get_win_preds(X_test)
    score_predictions = get_runs_preds(X_test)
    json_output = assemble_json(predictions, proba,score_predictions,test_df)
    return json_output

if __name__ == '__main__':
    main()