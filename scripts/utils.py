import pandas as pd
import numpy as np
import pickle

def get_movie_titles(movie_ids, data_path):
    df_movies = pd.read_csv(data_path)
    return df_movies[df_movies['movie_id'].isin(movie_ids)]['movie_title'].tolist()


def hybrid_recommendation(user_id, content_based_data_path, collaborative_model_path, cosine_sim_path, ratings_data_path, n=10, weight_content=0.5, weight_collab=0.5):
    # Load preprocessed data
    content_based_data = pd.read_csv(content_based_data_path)
    collaborative_filtering_data = pd.read_csv(ratings_data_path)
    
    # Load collaborative filtering model
    with open(collaborative_model_path, 'rb') as f:
        collaborative_filtering_model = pickle.load(f)

    # Load cosine similarity matrix
    cosine_sim = pd.read_pickle(cosine_sim_path).values

    # Get unseen movies
    user_ratings = collaborative_filtering_data[collaborative_filtering_data['user_id'] == user_id]['item_id'].unique()
    all_movies = collaborative_filtering_data['item_id'].unique()
    unrated_movies = [movie for movie in all_movies if movie not in user_ratings]
    
    # Collaborative filtering predictions
    collab_predictions = [collaborative_filtering_model.predict(user_id, movie_id).est for movie_id in unrated_movies]
    
    # Content-based predictions
    content_predictions = [np.mean([cosine_sim[movie_idx][content_based_data[content_based_data['movie_id'] == movie].index[0]]
                                   for movie_idx in user_ratings]) for movie in unrated_movies]
    
    # Hybrid prediction
    hybrid_predictions = [(weight_content * content_pred + weight_collab * collab_pred) 
                          for content_pred, collab_pred in zip(content_predictions, collab_predictions)]
    
    # Get top N recommended movies
    top_n_idx = np.argsort(hybrid_predictions)[-n:][::-1]
    top_n_movies = [unrated_movies[idx] for idx in top_n_idx]
    
    return top_n_movies

if __name__ == '__main__':
    recommendations = hybrid_recommendation(196, '../data/processed/content_based_data.csv', '../models/collaborative_filtering_model.pkl', '../models/cosine_similarity_matrix.pkl', '../data/processed/collaborative_filtering_data.csv')
    print(get_movie_titles(recommendations, '../data/processed/content_based_data.csv'))
