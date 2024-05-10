import requests
from datetime import timedelta, datetime
import time

def daterange(start_date):
    end_date = datetime.now() - timedelta(days=1)
    current_date = datetime(start_date.year, start_date.month, start_date.day)
    one_day = timedelta(days=1)
    while current_date <= end_date:
        if current_date.month not in [1, 2, 11, 12]:
            yield current_date
        current_date += one_day

def get_daily_game_ids(api_key, year, month, day, delay_seconds=1):
    ''' Fetches game IDs from the daily summary endpoint. '''
    try:
        # Ensure month and day are integers and format them to two digits
        month_str = f"{int(month):02d}"
        day_str = f"{int(day):02d}"
    except ValueError:
        raise ValueError("Month and Day must be convertible to integers.")

    # Construct the URL based on provided parameters
    url = f"https://api.sportradar.com/mlb/trial/v7/en/games/{year}/{month_str}/{day_str}/summary.json"
    
    # Set headers to define expected response format
    headers = {'Accept': 'application/json'}
    
    # Send the API request
    response = requests.get(url, params={'api_key': api_key}, headers=headers)
    
    # Debug print to check URL and response status
    print(f"Request URL: {url} | Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        games = data.get('league', {}).get('games', [])
        print(f"Games found on {year}-{month_str}-{day_str}: {len(games)}")
        # Wait for the specified delay time before making another request
        time.sleep(delay_seconds)
        return [game['game']['id'] for game in games]
    elif response.status_code >= 500:
        print("Server error, consider retrying the request")
        return []
    else:
        print(f"Failed to retrieve data: Status Code {response.status_code}")
        return []