import os
import pandas as pd
import scipy.spatial as sc
import sys
sys.path.append(os.path.abspath('import_song_data.py'))
import import_song_data


# use k-NN algorithm to find similar songs
def k_NN(song_row, k):
    distances = music_normalized.apply(lambda row: sc.distance.euclidean(row, song_row), axis=1)
    # store euclidean distance values with the song indices for easy access later on
    distance_frame = pd.DataFrame(data={'dist': distances, 'idx': distances.index})
    distance_frame = distance_frame.sort_values(by='dist')
    nearest = []
    for i in range(1, k):
        nearest.append(music_df.loc[int(distance_frame.iloc[i]['idx']), :])
    return nearest


music_df = import_song_data.read()
print(music_df)

# classify genres (terms) into numbers, with each genre getting a unique number
genres_class = {}
count = 0
for idx, row in music_df.iterrows():
    if row['terms'] not in genres_class.keys():
        genres_class[row['terms']] = count
        count += 1

music_df['terms'] = music_df.apply(lambda row: genres_class[row['terms']], axis=1)

# set list of numeric only columns for computing euclidean distances, ignoring year due to consistent missing values
numeric_cols = ['artist.hotttnesss', 'bars_start', 'beats_start', 'duration', 'start_of_fade_out', 'tatums_start',
                'tempo', 'terms', 'time_signature']

# the values are currently strings, so cast them to floats as well
music_numeric = music_df[numeric_cols].astype(float)

# normalize the dataset
music_normalized = music_numeric.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
print(music_normalized)

# calculate euclidean distances using a selected song
rand_song = music_normalized.loc[music_df['title'] == "Does It Float", :]
song = music_df.loc[music_df['title'] == "Does It Float", :]
nearest = k_NN(rand_song, 10)
print(song)
print(nearest)
