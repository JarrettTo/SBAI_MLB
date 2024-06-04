import sys
import os
from datetime import date
import json
from flask import Flask, render_template, request, jsonify
from flask import current_app
from functools import lru_cache
import threading
from flask_mysqldb import MySQL
import requests
import json
from datetime import datetime
root_dir = os.getenv('ROOT_DIR', 'default_fallback_path')

sys.path.append(root_dir)
from predictions import main

app = Flask(__name__)
app.config['MYSQL_HOST'] = '54.183.200.189'
app.config['MYSQL_PORT'] = 3307
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'jiustudios'
app.config['MYSQL_DB'] = 'sbai'
mysql = MySQL(app)

teamAbbMap = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW", 
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",  
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN"
}
def run_prediction():
    # This function will run the long process in the background.
    with app.app_context():
        preds = main()
        res= json.loads(preds)
        url = "https://sbai-com-frontend.vercel.app/api/mlb/schedule/today"
        today_games = None
        today_odds = None
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            today_games=response.json()  # Return the JSON from the response
        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)}), 500
        odds_url = "https://sbai-com-frontend.vercel.app/api/mlb/odds"
        try:
            response = requests.get(odds_url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            today_odds=response.json()  # Return the JSON from the response
        except requests.exceptions.RequestException as e:
            return jsonify({"error": str(e)}), 500
        for game in today_games:
            gameId= None
            print("GAME:", game["homeTeam"])
            print("ABBR:", teamAbbMap[game["homeTeam"]])
            for pred in res:
                print("PRED:", pred["home_team"])
                if teamAbbMap[game["homeTeam"]] == pred["home_team"] and teamAbbMap[game["awayTeam"]] == pred["away_team"] :
                    gameId =  pred["id"]
                    break
            print("GAME ID:", gameId)
            cursor = mysql.connection.cursor()
            if game["schedule"].endswith('Z'):
                game["schedule"] = game["schedule"][:-1] + '+00:00'  # Properly handle UTC offset

            # Convert ISO 8601 string to datetime object
            datetime_object = datetime.fromisoformat(game["schedule"])

            # Format this datetime object into a MySQL-compatible string
            mysql_format_datetime = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(''' INSERT IGNORE INTO MLB_Games (id, home_team, away_team, schedule, location) VALUES(%s,%s, %s, %s, %s)''',(gameId,game["homeTeam"], game["awayTeam"],mysql_format_datetime, game["arena"]))
            mysql.connection.commit()
        for pred in res:
            cursor = mysql.connection.cursor()
            cursor.execute(''' INSERT IGNORE INTO MLB_Predictions VALUES(%s,%s, %s, %s, %s)''',(pred["id"],pred["ml_pred"], float(pred["ml_conf"]),float(pred["ou_pred"]), int(pred["ou_conf"])))
            mysql.connection.commit()
        for game in today_odds:
            gameId= None
            for pred in res:
                if teamAbbMap[game["home_team"]] == pred["home_team"] and teamAbbMap[game["away_team"]] == pred["away_team"] :
                    gameId =  pred["id"]
            cursor = mysql.connection.cursor()
            print("PRICE:", game["bookmakers"][0]["markets"][1]["outcomes"][0]["price"])
            print("NAME:", game["bookmakers"][0]["markets"][2]["outcomes"][0]["name"])
            cursor.execute(''' INSERT IGNORE INTO MLB_Odds VALUES(%s,%s, %s, %s, %s,%s,%s, %s,%s, %s)''',(gameId,"draftkings",
                                                                                            game["bookmakers"][0]["markets"][1]["outcomes"][0]["price"], 
                                                                                            game["bookmakers"][0]["markets"][2]["outcomes"][0]["name"], 
                                                                                            game["bookmakers"][0]["markets"][2]["outcomes"][0]["price"],
                                                                                            game["bookmakers"][0]["markets"][0]["outcomes"][0]["price"],
                                                                                            game["bookmakers"][0]["markets"][1]["outcomes"][1]["price"], 
                                                                                            game["bookmakers"][0]["markets"][2]["outcomes"][1]["name"], 
                                                                                            game["bookmakers"][0]["markets"][2]["outcomes"][1]["price"],
                                                                                            game["bookmakers"][0]["markets"][0]["outcomes"][1]["price"]
                                                                                            ))
            mysql.connection.commit()
        
        cursor.close()
        print("Prediction complete:", res)
@app.route("/")
def index():
    return "hello"

@app.route("/predict") 
def predict():
    thread = threading.Thread(target=run_prediction)
    thread.start()
    # Immediately respond to the request indicating that processing has started.
    return "Prediction process started!"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=4000)
