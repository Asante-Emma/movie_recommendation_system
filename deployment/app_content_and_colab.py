import gradio as gr
import pickle
import pandas as pd

# Load models and data
with open('../models/collaborative_filtering_model.pkl', 'rb') as f:
    collaborative_filtering_model = pickle.load(f)

cosine_sim = pd.read_pickle('../models/cosine_similarity_matrix.pkl')

# Load preprocessed data
movies_data = pd.read_csv('../data/processed/content_based_data.csv')
collaborative_filtering_data = pd.read_csv('../data/processed/collaborative_filtering_data.csv')

# Function for content-based recommendation
def recommend_content_based(movie_title, n=10):
    movie_idx = movies_data[movies_data['movie_title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_n_movies = [movies_data['movie_title'][i[0]] for i in sim_scores[1:n+1]]
    return top_n_movies

# Function for collaborative filtering recommendation
def recommend_collaborative(user_id, n=10):
    user_ratings = collaborative_filtering_data[collaborative_filtering_data['user_id'] == user_id]['item_id'].unique()
    movie_ids = collaborative_filtering_data['item_id'].unique()
    unrated_movies = [movie for movie in movie_ids if movie not in user_ratings]
    predictions = [collaborative_filtering_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    recommended_movie_ids = [pred.iid for pred in top_n]
    recommended_movies = movies_data[movies_data['movie_id'].isin(recommended_movie_ids)]['movie_title'].tolist()
    return recommended_movies

# Gradio Interface
def hybrid_recommendation(user_id, movie_title, n=10):
    collaborative_recs = recommend_collaborative(user_id, n)
    content_based_recs = recommend_content_based(movie_title, n)
    return {
        'Collaborative Filtering Recommendations': collaborative_recs,
        'Content-Based Recommendations': content_based_recs
    }

# Define inputs and outputs for Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Collaborative Filtering & Content-Based Movie Recommendation System")

    with gr.Row():
        user_id_input = gr.Number(label="User ID", value=1)
        movie_title_input = gr.Textbox(label="Movie Title", placeholder="Enter a movie title")

    with gr.Row():
        num_recommendations = gr.Number(label="Number of Recommendations", value=10)

    recommendation_button = gr.Button("Get Recommendations")
    recommendation_output = gr.JSON(label="Recommendations")

    # Bind function to Gradio interface
    recommendation_button.click(
        hybrid_recommendation,
        inputs=[user_id_input, movie_title_input, num_recommendations],
        outputs=recommendation_output
    )

# Launch the interface
demo.launch(share=True)
