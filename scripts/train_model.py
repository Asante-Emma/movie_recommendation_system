import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_collaborative_filtering_model(data_path, model_output_dir):
    # Load data
    df_ratings = pd.read_csv(data_path)
    
    # Prepare data for Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)
    
    # Split train/test
    trainset = data.build_full_trainset()
    
    # Train the KNNBasic model
    sim_options = {
        'name': 'pearson_baseline',
        'user_based': True
    }
    algo = KNNBasic(k=35, min_k=2, sim_options=sim_options)
    algo.fit(trainset)
    
    # Save the model
    os.makedirs(model_output_dir, exist_ok=True)
    model_file = os.path.join(model_output_dir, 'collaborative_filtering_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(algo, f)

    print(f"Collaborative Filtering model saved at {model_file}")


def train_content_based_model(data_path, model_output_dir):
    # Load data
    df_movies = pd.read_csv(data_path)

    # Feature extraction using TF-IDF on genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movies['genres'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the similarity matrix
    os.makedirs(model_output_dir, exist_ok=True)
    cosine_sim_file = os.path.join(model_output_dir, 'cosine_similarity_matrix.pkl')
    pd.DataFrame(cosine_sim).to_pickle(cosine_sim_file)

    print(f"Content-Based model (cosine similarity matrix) saved at {cosine_sim_file}")


if __name__ == "__main__":
    train_collaborative_filtering_model(
        data_path='../data/processed/collaborative_filtering_data.csv',
        model_output_dir='../models'
    )
    
    train_content_based_model(
        data_path='../data/processed/content_based_data.csv',
        model_output_dir='../models'
    )
