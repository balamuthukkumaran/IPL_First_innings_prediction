import pandas as pd
import glob

csv_files = glob.glob('/Users/balamuthukkumaranrajan/Downloads/ipl_csv2 (1) 2/*.csv')
dfs = []

for file in csv_files:
    
    df = pd.read_csv(file)
    first_innings_df = df[df['innings'] == 1]
    new_df = pd.DataFrame(columns=['date', 'venue', 'bat_team', 'bowl_team', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total_score', 'run_rate'])

    total_runs = 0
    total_wickets = 0

    for _, row in first_innings_df.iterrows():

        date = row['start_date']
        venue = row['venue']
        bat_team = row['batting_team']
        bowl_team = row['bowling_team']
        runs = total_runs + row['runs_off_bat'] + row['extras']
        wickets = total_wickets
        if pd.notnull(row['wicket_type']):
            wickets += 1

        overs = row['ball']

        last_5_balls_df = first_innings_df[(first_innings_df['ball'] <= overs) & (first_innings_df['ball'] > overs - 0.5)]
        runs_last_5 = last_5_balls_df['runs_off_bat'].sum() + last_5_balls_df['extras'].sum()
        wickets_last_5 = last_5_balls_df['wicket_type'].count()
        total_score = first_innings_df['runs_off_bat'].sum() + first_innings_df['extras'].sum()
        run_rate = total_runs / overs
        total_runs = runs
        total_wickets = wickets

        new_df = new_df.append({'date': date, 'venue': venue, 'bat_team': bat_team, 'bowl_team': bowl_team, 'runs': runs, 'wickets': wickets, 'overs': overs, 'runs_last_5': runs_last_5, 'wickets_last_5': wickets_last_5, 'total_score': total_score, 'run_rate': run_rate}, ignore_index=True)

    dfs.append(new_df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv('combined_first_innings_runrate.csv', index=False)