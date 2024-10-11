import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Load models and data
with open('../models/collaborative_filtering_model.pkl', 'rb') as f:
    collaborative_filtering_model = pickle.load(f)

cosine_sim = pd.read_pickle('../models/cosine_similarity_matrix.pkl')
collaborative_filtering_data = pd.read_csv('../data/processed/collaborative_filtering_data.csv')
content_based_data = pd.read_csv('../data/processed/content_based_data.csv')

# Get movie titles from recommendations
def get_movie_titles(movie_ids):
    return content_based_data[content_based_data['movie_id'].isin(movie_ids)]['movie_title'].tolist()

# Hybrid recommendation function
def hybrid_recommendation(user_id, n=10, weight_content=0.5, weight_collab=0.5):
    # Get movies the user has already rated
    user_ratings = collaborative_filtering_data[collaborative_filtering_data['user_id'] == user_id]['item_id'].unique()
    all_movies = collaborative_filtering_data['item_id'].unique()

    # Get unseen movies
    unrated_movies = [movie for movie in all_movies if movie not in user_ratings]
    
    # Predict ratings using collaborative filtering
    collab_predictions = [collaborative_filtering_model.predict(user_id, movie_id).est for movie_id in unrated_movies]

    # Get content-based recommendations using precomputed cosine similarity
    content_predictions = []
    for movie in unrated_movies:
        sim_scores = [cosine_sim[movie_idx][content_based_data[content_based_data['movie_id'] == movie].index[0]] 
                      for movie_idx in user_ratings if movie in content_based_data['movie_id'].values]
        content_predictions.append(np.mean(sim_scores) if sim_scores else 0)

    # Hybrid prediction: weighted average of both predictions
    hybrid_predictions = [(weight_content * content_pred + weight_collab * collab_pred) 
                          for content_pred, collab_pred in zip(content_predictions, collab_predictions)]
    
    # Get top N recommended movies
    top_n_idx = np.argsort(hybrid_predictions)[-n:][::-1]
    top_n_movies = [unrated_movies[idx] for idx in top_n_idx]
    
    # Return movie titles instead of IDs
    return get_movie_titles(top_n_movies)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Hybrid Movie Recommendation System")

    with gr.Row():
        user_id_input = gr.Number(label="User ID", value=1)
        num_recommendations = gr.Number(label="Number of Recommendations", value=10)
        weight_content_input = gr.Number(label="Weight for Content-Based", value=0.5, step=0.1)
        weight_collab_input = gr.Number(label="Weight for Collaborative", value=0.5, step=0.1)

    recommendation_button = gr.Button("Get Recommendations")
    recommendation_output = gr.JSON(label="Recommendations")

    # Bind function to Gradio interface
    recommendation_button.click(
        hybrid_recommendation,
        inputs=[user_id_input, num_recommendations, weight_content_input, weight_collab_input],
        outputs=recommendation_output
    )

# Launch the interface
demo.launch(share=True)