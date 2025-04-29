import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Create a mock dataset of songs
data = {
    "song_title": ["Blinding Lights", "Shape of You", "Bohemian Rhapsody", "Dancing Queen", "Sweet Child O' Mine", 
                   "Uptown Funk", "Billie Jean", "Hotel California", "Rolling in the Deep", "Smells Like Teen Spirit"],
    "artist": ["The Weeknd", "Ed Sheeran", "Queen", "ABBA", "Guns N' Roses", 
               "Bruno Mars", "Michael Jackson", "Eagles", "Adele", "Nirvana"],
    "genre": ["Pop", "Pop", "Rock", "Disco", "Rock", 
              "Funk", "Pop", "Rock", "Soul", "Rock"],
    "tempo": [171, 96, 145, 100, 125, 
              115, 117, 147, 105, 116],
    "energy": [0.73, 0.65, 0.60, 0.88, 0.90, 
               0.85, 0.70, 0.55, 0.78, 0.92],
    "danceability": [0.51, 0.83, 0.40, 0.67, 0.45, 
                     0.86, 0.91, 0.58, 0.63, 0.49]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Define numerical and categorical columns
numerical_cols = ["tempo", "energy", "danceability"]
categorical_cols = ["genre", "artist"]

# Create a ColumnTransformer to preprocess numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),  # Scale numerical features
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols)  # Encode categorical features
    ]
)

# Fit and transform the data
X = preprocessor.fit_transform(df)

# Step 3: Train the KNN model
# Use NearestNeighbors to find similar songs
knn = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
knn.fit(X)

# Step 4: Function to recommend songs
def recommend_songs(song_title, df, X, knn, num_recommendations=5):
    # Find the index of the input song
    song_idx = df.index[df["song_title"] == song_title].tolist()
    if not song_idx:
        raise ValueError(f"Song '{song_title}' not found in the dataset.")
    song_idx = song_idx[0]

    # Find the nearest neighbors (excluding the song itself)
    distances, indices = knn.kneighbors([X[song_idx]], n_neighbors=num_recommendations + 1)
    indices = indices[0][1:]  # Exclude the first index (the song itself)
    distances = distances[0][1:]

    # Collect recommendations
    recommendations = []
    print(f"\nRecommendations for '{song_title}':")
    for rank, idx in enumerate(indices, 1):
        recommended_song = df.iloc[idx]
        recommendations.append({
            "Rank": rank,
            "Song Title": recommended_song["song_title"],
            "Artist": recommended_song["artist"],
            "Genre": recommended_song["genre"],
            "Distance": distances[rank - 1]
        })
        print(f"{rank}. {recommended_song['song_title']} by {recommended_song['artist']} (Genre: {recommended_song['genre']}, Distance: {distances[rank - 1]:.4f})")

    return recommendations

# Step 5: Generate recommendations for a sample song
song_to_recommend = "Blinding Lights"
try:
    recommendations = recommend_songs(song_to_recommend, df, X, knn, num_recommendations=5)

    # Step 6: Save recommendations to a CSV file
    if recommendations:
        df_recommendations = pd.DataFrame(recommendations)
        df_recommendations.to_csv("music_recommendations.csv", index=False)
        print(f"\nSaved recommendations to music_recommendations.csv")
    else:
        print("\nNo recommendations generated.")
except Exception as e:
    print(f"Error generating recommendations: {e}")
