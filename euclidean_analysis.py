import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.spatial.distance import euclidean

player_data = pd.read_csv("weekly_points_data.csv")
print(player_data.head())

#replacing values in datafram so it can be used for euclidean distance comparison
player_data.replace("-", 0, inplace=True)
player_data.replace("BYE", 0, inplace=True)
player_data.fillna(0, inplace=True)

#position values
position_dict = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 4,
    "DST": 5,
    "K": 6
}

player_data['Pos_Num'] = player_data['Pos'].map(position_dict)

#team values
team_dict = {
    "WAS": 1, "DAL": 2, "PHI": 3, "NYG": 4,
    "CHI": 5, "DET": 6, "MIN": 7, "GB": 8,
    "NO": 9, "ATL": 10, "TB": 11, "CAR": 12,
    "SF": 13, "LAR": 14, "ARI": 15, "SEA": 16,
    "BAL": 17, "PIT": 18, "CIN": 19, "CLE": 20,
    "BUF": 21, "NE": 22, "MIA": 23, "NYJ": 24,
    "JAC": 25, "HOU": 26, "TEN": 27, "IND": 28,
    "KC": 29, "DEN": 30, "LAC": 31, "LV": 32,
    "FA": 33
}

player_data['Team_Num'] = player_data['Team'].map(team_dict)

print(player_data.head())

# normalizes the data with the given numerical columns to use in eucildean distance
normalize = MinMaxScaler()
player_data_scale = normalize.fit_transform(player_data[['Pos_Num', 'Team_Num', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 
           'Week 6',  'Week 7', 'Week 8', 'Week 9', 'Week 10', 'Week 11', 
           'Week 12', 'Week 13', 'Week 14', 'Week 15', 'Week 16', 'Week 17', 
           'Week 18', 'AVG', 'TTL']])

#I decided to use knn as it made it easier to get the code to work.
#n neigbors sets it to the 10 closest neighbors, uses 11 since the cloests will be
#the actual player
knn = NearestNeighbors(n_neighbors=11, algorithm='auto').fit(player_data_scale)

#computes the eucildean distance for the given player 
def find_ten_closest(player_name):

    player_index = player_data[player_data['Player'] == player_name].index[0]
    distance, index = knn.kneighbors([player_data_scale[player_index]])

    similar_player = player_data.iloc[index[0][1:]]
    similar_player_stats = similar_player[['Player', 'Team', 'Pos', 'AVG', "TTL"]]

    print(f"{player_name}'s Data:")
    columns_to_print = ["Player", "Team", "Pos", "AVG", "TTL"]
    print(player_data.loc[player_data['Player'] == player_name, columns_to_print])

    print(f"Top 10 Similar Players to {player_name}:")

    print(similar_player_stats)
    print(f"The distances for {player_name}: {distance}")

find_ten_closest("Patrick Mahomes II")
find_ten_closest("Terry McLaurin")
find_ten_closest("Rachaad White")