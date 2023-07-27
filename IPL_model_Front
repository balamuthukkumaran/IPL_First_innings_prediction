
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('/Users/balamuthukkumaranrajan/combined_first_innings_runrate.csv')

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad']

df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

df = df[df['overs'] >= 5.0]

df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%y'))

df = df[df['date'].dt.year <= 2022]

encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team','venue'])

encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals',
                         'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
                         'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                         'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                         'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Capitals',
                         'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
                         'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                         'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
                         'venue_Arun Jaitley Stadium', 'venue_Barabati Stadium',
                         'venue_Brabourne Stadium', 'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
                         'venue_Dr DY Patil Sports Academy', 'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
                         'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens', 'venue_Feroz Shah Kotla',
                         'venue_Himachal Pradesh Cricket Association Stadium', 'venue_Holkar Cricket Stadium',
                         'venue_JSCA International Stadium Complex', 'venue_Kingsmead', 'venue_M Chinnaswamy Stadium',
                         'venue_MA Chidambaram Stadium', 'venue_Maharashtra Cricket Association Stadium',
                         'venue_Narendra Modi Stadium', 'venue_New Wanderers Stadium', 'venue_Newlands',
                         'venue_Punjab Cricket Association Stadium', 'venue_Rajiv Gandhi International Stadium',
                         'venue_Sardar Patel Stadium', 'venue_Sawai Mansingh Stadium',
                         'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
                         'venue_St George\'s Park', 
                         'venue_SuperSport Park', 
                         'venue_Wankhede Stadium', 'venue_Zayed Cricket Stadium',
                         'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5',
                         'total_score', 'run_rate']]

X_train = encoded_df.drop(labels='total_score', axis=1)[encoded_df['date'].dt.year <= 2021]
X_test = encoded_df.drop(labels='total_score', axis=1)[encoded_df['date'].dt.year >= 2022]

y_train = encoded_df[encoded_df['date'].dt.year <= 2021]['total_score'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2022]['total_score'].values

X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor.fit(X_train, y_train)

models = [('Linear Regression', regressor), ('Random Forest', rf_regressor), ('Gradient Boosting', gb_regressor)]
best_model_name = ''
best_model_mse = float('inf')
best_model_mae = float('inf')

for model_name, model in models:

    filename = 'first-innings-score-{}.pkl'.format(model_name.lower().replace(' ', '-'))
    pickle.dump(model, open(filename, 'wb'))

    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))

    if mse < best_model_mse:
        best_model_mse = mse
        best_model_name = model_name
        best_model_filename = filename
    
    if mae < best_model_mae:
        best_model_mae = mae

app = Flask(__name__)
best_model = pickle.load(open(best_model_filename, 'rb'))

def get_team_encoding(team):
    teams = ['Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab', 'Kolkata Knight Riders',
             'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
    
    encoding = [0] * len(teams)
    if team in teams:
        encoding[teams.index(team)] = 1
    return encoding

def get_venue_encoding(venue):
    
    venues = ['Arun Jaitley Stadium', 'Barabati Stadium', 
              'Brabourne Stadium', 'Buffalo Park', 'De Beers Diamond Oval', 'Dr DY Patil Sports Academy', 
              'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Dubai International Cricket Stadium', 
              'Eden Gardens', 'Feroz Shah Kotla', 'Himachal Pradesh Cricket Association Stadium', 
              'Holkar Cricket Stadium', 'JSCA International Stadium Complex', 'Kingsmead', 'M Chinnaswamy Stadium', 
              'MA Chidambaram Stadium', 'Maharashtra Cricket Association Stadium', 'Narendra Modi Stadium',  
              'New Wanderers Stadium', 'Newlands', 'Punjab Cricket Association Stadium', 
              'Rajiv Gandhi International Stadium', 'Sardar Patel Stadium', 'Sawai Mansingh Stadium', 'Sharjah Cricket Stadium', 
              'Sheikh Zayed Stadium', 'St George\'s Park','SuperSport Park', 'Wankhede Stadium', 'Zayed Cricket Stadium']  
    
    encoding = [0] * len(venues)

    if venue in venues:
        encoding[venues.index(venue)] = 1
    return encoding

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=['GET','POST'])

def predict():

    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    overs = float(request.form['overs'])
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    runs_in_prev_5 = int(request.form['runs_in_prev_5'])
    wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
    venue = request.form['venue']

    temp_array = []
    temp_array += get_team_encoding(batting_team)
    temp_array += get_team_encoding(bowling_team)
    temp_array += get_venue_encoding(venue)

    run_rate = runs / overs
    temp_array += [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5, run_rate]
    temp_array = np.array([temp_array])

    predicted_score = int(best_model.predict(temp_array)[0])
    final_score_range = (predicted_score-5, predicted_score+5)
    run_rate_predicted_range = (round((predicted_score/20)-0.4, 1), round((predicted_score/20)+0.4, 1))

    return render_template("result.html", predicted_score=predicted_score, final_score_range=final_score_range,
                           run_rate_predicted_range=run_rate_predicted_range)

if __name__ == "__main__":
    app.run(debug=True)
