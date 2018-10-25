"""
    import_song_data.py -- Prepares the million song dataset
    The entire dataset is 280 GB, so we'll use the 1.8 GB subset they provide
    The data subset contains 10,000 songs, each of which includes the following fields
        * artist_name
        * bars_start: shape = (99,) the number of bars a song has (99 in this example)
        * beats_start: shape = (397,) number of beats a song has
        * danceability: danceability measure of this song (according to the.echonest.org) 0 => not analyzed
        * duration (in seconds)
        * end_of_fade_in: time til end of fade-in at beginning of song (in seconds)
        * start_of_fade_out: start to of fade-out in seconds
        * energy: energy measure between 0 and 1 (0 => not analyzed)
        * key: estimation of the key the song is in
        * key_confidence: confidence of key estimation between 0 and 1
        * loudness: general loudness of song in dB
        * mode: estimation of the mode of the song (e.g. Lydian, Dorian, Aeolian, etc.)
        * release: album name from which the track was taken (always just one album)
        * sections_start: shape = (10,) start time of each section (verse, chorus, etc.) (this song has 10)
        * similar_artists: shape(100,) list of 100 similar artists according to the.echonest.org
        * song_hotttness: popularity of song from 0 to 1 based on when dataset is downloaded
        * tempo: tempo in BPM
        * time_signature: time signature in usual number of beats/bar
        * time_signature_confidence: confidence of signature value from 0 to 1
        * title
        * year
    There are many more fields, but we'll focus on these

    Dataset provided by:
        Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere.
        The Million Song Dataset. In Proceedings of the 12th International Society
        for Music Information Retrieval Conference (ISMIR 2011), 2011.
"""

import os
import sys
import time
import glob
import numpy as np
import hdf5_getters
import pandas as pd
import scipy.spatial as sc

# path to uncompressed song subset -- adjust to your local configuration
msd_path = 'D:\Programming\Python\MillionsongSubset'
msd_data_path = os.path.join(msd_path, 'data')
msd_addf_path = os.path.join(msd_path, 'AdditionalFiles')


# iterate over all files in all subdirectories
def apply_to_all_files(basedir, func=lambda x: x, ext='.h5'):
    '''
    :param basedir: base directory of the dataset
    :param func: function to apply to all filenames
    :param ext: extension, .h5 by default
    '''
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*'+ext))
        # count files
        for f in files:
            func(f)


# use k-NN algorithm to find similar songs
def one_NN(song_row):
    distances = music_normalized.apply(lambda row: sc.distance.euclidean(row, song_row), axis=1)
    distance_frame = pd.DataFrame(data={'dist': distances, 'idx': distances.index})
    distance_frame = distance_frame.sort_values(by='dist')
    nearest = music_df.loc[int(distance_frame.iloc[1]['idx']), :]
    return nearest

# print out all song file locations
# apply_to_all_files(msd_data_path, func=lambda x: print(x))


# use hdf5_getters to get attributes of an artist
h5 = hdf5_getters.open_h5_file_read('D:\Programming\Python\MillionsongSubset\data\A\A\A\TRAAAAW128F429D538.h5')
artist = hdf5_getters.get_artist_name(h5)
title = hdf5_getters.get_title(h5)
year = hdf5_getters.get_year(h5)
num_songs = hdf5_getters.get_num_songs(h5)
familiarity = hdf5_getters.get_artist_familiarity(h5)
hotness = hdf5_getters.get_song_hotness(h5)
similar_artists = np.array(hdf5_getters.get_similar_artists(h5))
print('Artist:', artist, '\nTitle:', title, '\nNums songs:', num_songs, '\nArtist familiarity:', familiarity,
      '\nHotness:', hotness, '\nSimilar artists:', similar_artists, '\nYear:', year)

# read the dataset as a csv rather than getting attributes via hdf5_getters
music_df = pd.read_csv('music.csv', index_col=False, na_values='?', delimiter=',', header=None, engine='python')
# set the column labels to equal the values in the 1st row
music_df.columns = music_df.iloc[0]
music_df = music_df.reindex(music_df.index.drop(0))
print(music_df)

# classify genres (terms) into numbers
genres_class = {}
count = 0
for idx, row in music_df.iterrows():
    if row['terms'] not in genres_class.keys():
        genres_class[row['terms']] = count
        count += 1

music_df['terms'] = music_df.apply(lambda row: genres_class[row['terms']], axis=1)

# set list of numeric only columns for computing euclidean distances
numeric_cols = ['artist.hotttnesss', 'bars_start', 'beats_start', 'duration', 'start_of_fade_out', 'tatums_start',
                'tempo', 'terms', 'time_signature', 'year']

music_numeric = music_df[numeric_cols].astype(float)
print(music_numeric)

# normalize the dataset
music_normalized = music_numeric.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
print(music_normalized)

# calculate euclidean distances using a selected song
rand_song = music_normalized.loc[music_df['title'] == "Does It Float", :]
song = music_df.loc[music_df['title'] == "Does It Float", :]
nearest = one_NN(rand_song)
print(song)
print(nearest)

h5.close()
