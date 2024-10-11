import os
import pandas as pd

def load_and_preprocess_data(movies_path, ratings_path, output_dir):
    # Load data
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)

    # Preprocessing movies
    df_movies['genres'] = df_movies['genres'].fillna('')  # Fill missing genres

    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    
    content_based_data = df_movies[['movie_id', 'movie_title', 'genres']]
    collaborative_filtering_data = df_ratings

    content_based_file = os.path.join(output_dir, 'content_based_data.csv')
    collaborative_filtering_file = os.path.join(output_dir, 'collaborative_filtering_data.csv')

    content_based_data.to_csv(content_based_file, index=False)
    collaborative_filtering_data.to_csv(collaborative_filtering_file, index=False)

    print(f"Data preprocessed and saved in {output_dir}")
