import pandas as pd

columns = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
           'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=columns)

def extract_genres(row):
    genres = []
    for genre in columns[5:]:
        if row[genre] == 1:
            genres.append(genre)
    return '|'.join(genres)

df['genres'] = df.apply(extract_genres, axis=1)

movies_df = df[['movieId', 'title', 'genres']]

movies_df.to_csv('movies.csv', index=False)

print(" movies.csv file created successfully.")
