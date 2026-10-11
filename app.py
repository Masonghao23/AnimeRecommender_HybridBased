import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Anime Recommender System",
    page_icon="🎌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Hide Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    .stToolbar {display: none;}
    .stActionButton {display: none;}
    
    /* Hide the hamburger menu */
    .css-14xtw13.e8zbici0 {display: none;}
    
    /* Hide "Made with Streamlit" footer */
    .css-h5rgaw.egzxvld1 {display: none;}
    
    /* Hide top right menu and icons */
    .css-1dp5vir.e8zbici0 {display: none;}
    
    /* Hide GitHub icon and other top-right elements */
    .css-1rs6os.edgvbvh3 {display: none;}
    
    /* Hide settings menu */
    .css-1kyxreq.etr89bj2 {display: none;}
    
    /* Hide view source button */
    .viewerBadge_container__1QSob {display: none;}
    
    /* Hide streamlit branding */
    .css-1v0mbdj.etr89bj1 {display: none;}
    
    /* Additional cleanup */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
        font-size: 1.1rem;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .anime-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .score-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .synopsis {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess all datasets"""
    try:
        anime = pd.read_csv('anime.csv', low_memory=False)
        rating = pd.read_csv('rating_complete.csv', low_memory=False)
        synopsis = pd.read_csv('anime_with_synopsis.csv', low_memory=False)
        animelist = pd.read_csv('animelist.csv', low_memory=False)
        
        anime = anime.rename(columns={'MAL_ID': 'anime_id'})
        synopsis = synopsis.rename(columns={'MAL_ID': 'anime_id'})
        
        return anime, rating, synopsis, animelist
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None

def preprocess_data(anime, rating, synopsis, animelist):
    """Preprocess data exactly as in the notebook"""
    # Merge anime data with synopsis
    anime_data = pd.merge(anime, synopsis.drop(columns=['Name', 'Score', 'Genres']), on='anime_id')
    anime_data['Score'] = anime_data['Score'].replace('Unknown', 0).astype(float)
    
    # Filter animelist - remove unwanted statuses
    animelist = animelist.drop(animelist[animelist['watching_status'].isin([0, 5, 33, 55])].index)
    
    # Create status weight mapping
    status_weight_map = {
        1: 0.8,  # Currently Watching
        2: 1.0,  # Completed
        3: 0.3,  # On Hold
        4: -0.5, # Dropped
        6: 0.2   # Plan to watch
    }
    
    anime_status = animelist[['anime_id', 'watching_status']].copy()
    anime_status['status_score'] = anime_status['watching_status'].map(status_weight_map)
    
    # Create anime rating aggregation
    anime_rating = pd.DataFrame(rating.groupby('anime_id').agg(
        rating=('rating', 'mean'), 
        vote=('rating', 'sum')
    )).reset_index()
    anime_rating['rating'] = anime_rating['rating'].fillna(0)
    
    # Filter by vote threshold (85th percentile)
    m = anime_rating['vote'].quantile(0.9)
    qualified_anime = anime_rating.copy().loc[anime_rating['vote'] >= m]
    rating_filtered = rating[rating['anime_id'].isin(qualified_anime['anime_id'])]
        
    # Create user-anime matrix
    user_anime_matrix = rating_filtered.pivot_table(
        index='user_id', 
        columns='anime_id', 
        values='rating'
    ).fillna(0).astype('float32')
    
    return anime_data, anime_status, user_anime_matrix, synopsis

@st.cache_data
def create_content_similarity(anime_data):
    """Create TF-IDF matrix and cosine similarity"""
    tfidf = TfidfVectorizer(stop_words='english')
    anime_data['sypnopsis'] = anime_data['sypnopsis'].fillna('')
    tfidf_matrix = tfidf.fit_transform(anime_data['sypnopsis'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

def get_recommendations(anime_name, anime, synopsis, user_anime_matrix, anime_status, cosine_sim):
    """Get recommendations"""
    try:
        # Get anime_id
        anime_id = anime[anime['Name'] == anime_name]['anime_id'].values[0]
        
        # Collaborative Filtering
        users_ratings = user_anime_matrix[anime_id]
        cf_scores = user_anime_matrix.corrwith(users_ratings)
        cf_scores = pd.DataFrame(cf_scores, columns=['cf_score']).dropna()
        
        # Content Based Filtering
        anime_index = synopsis[synopsis['anime_id'] == anime_id].index[0]
        cb_scores = pd.Series(cosine_sim[anime_index], index=synopsis['anime_id'])
        cb_scores = pd.DataFrame(cb_scores, columns=['cb_score'])
        
        # Status-based scoring
        ws_scores = anime_status.groupby('anime_id')['status_score'].mean().reset_index()
        
        # Merge all scores
        scores = cf_scores.merge(cb_scores, on='anime_id').fillna(0)
        scores = scores.merge(ws_scores[['anime_id', 'status_score']], on='anime_id').fillna(0)
        
        # Calculate final score
        scores['final_score'] = 0.6 * scores['cf_score'] + 0.3 * scores['cb_score'] + 0.1 * scores['status_score']
        
        # Merge with anime names and synopsis
        df_name = anime[['anime_id', 'Name']]
        df_synopsis = synopsis[['anime_id', 'sypnopsis']]
        result = scores.merge(df_name, on='anime_id')
        result = result.merge(df_synopsis, on='anime_id')
        
        # Return top 10 recommendations
        return result.sort_values('final_score', ascending=False).head(10)
        
    except (IndexError, KeyError) as e:
        st.error(f"Anime name '{anime_name}' not found or not eligible for recommendation.")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎌 Anime Recommender System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover recommended anime by hybrid algorithm</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading anime database...'):
        anime, rating, synopsis, animelist = load_data()
    
    if anime is None:
        st.error("Failed to load data files. Please ensure all CSV files are in the same directory.")
        return
    
    # Preprocess data
    with st.spinner('Preprocessing data...'):
        anime_data, anime_status, user_anime_matrix, synopsis = preprocess_data(anime, rating, synopsis, animelist)
        cosine_sim = create_content_similarity(anime_data)
    
    # Create selectbox for anime selection
    st.markdown("### Select an anime to get recommendations:")
    
    # Get list of available anime names
    available_anime = sorted(anime['Name'].dropna().unique())
    
    selected_anime = st.selectbox(
        "Choose an anime:",
        options=available_anime,
        index=None,
        placeholder="Type or select an anime name..."
    )
    
    # Get recommendations button
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        if selected_anime:
            with st.spinner('Generating recommendations...'):
                recommendations = get_recommendations(
                    selected_anime, anime, synopsis, user_anime_matrix, anime_status, cosine_sim
                )
            
            if recommendations is not None:
                st.markdown("### 🎯 Recommended Anime for You:")
                
                for idx, row in recommendations.iterrows():
                    # Create recommendation card
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="anime-title">{row['Name']}</div>
                        <span class="score-badge">Match Score: {row['final_score']:.3f}</span>
                        <div class="synopsis">{row['sypnopsis'][:200]}{'...' if len(str(row['sypnopsis'])) > 200 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show detailed scores in expander
                with st.expander("📊 View Detailed Scores"):
                    st.dataframe(
                        recommendations[['Name', 'cf_score', 'cb_score', 'status_score', 'final_score']].round(4),
                        use_container_width=True
                    )
        else:
            st.warning("Please select an anime first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Hybrid Recommendation Algorithm (Collaborative Filtering + Content Based)"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
