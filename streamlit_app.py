import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import json
import os

# Page configuration with British English spelling
st.set_page_config(
    page_title="LLM Word Relationships Visualiser", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1E3A8A;}
    .stTabs > div > button {
        font-size: 1.1rem;
        padding: 0.8rem 1.5rem;
        border: 2px solid #1E3A8A;
        border-radius: 8px;
        margin: 0 0.5rem;
        transition: all 0.3s ease;
    }
    .stTabs > div > button:hover {
        background-color: #3B82F6;
        color: white;
        transform: translateY(-2px);
    }
    .stTabs > div > button[aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
        border-color: #1E3A8A;
        font-weight: bold;
    }
    .stButton>button {background-color: #1E3A8A; color: white;}
    .stButton>button:hover {background-color: #3B82F6;}
    .st-emotion-cache-16txtl3 h1 {font-size: 2.5rem; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.title("Exploring Word Relationships in Language Models")

with st.expander("About this application", expanded=True):
    st.markdown("""
    This application visualises how language models understand relationships between words by representing them in 3D space. 
    
    Words that appear in similar contexts or have related meanings will be positioned closer together in this space. This is a simplified 
    representation of the high-dimensional "word embeddings" that language models use to understand language.
    
    **Try it out:**
    1. Enter words in the sidebar or use the defaults
    2. Explore the 3D space by rotating, zooming, and panning
    3. Show word relationships to visualise semantic connections
    4. Try the analogy feature to see how the model completes patterns like "king is to man as queen is to woman"
    """)

# Pre-computed GloVe embeddings
@st.cache_data
def load_embeddings():
    # Check if pre-computed embeddings exist
    if os.path.exists('glove_embeddings.csv'):
        try:
            # Load from CSV
            df = pd.read_csv('glove_embeddings.csv', index_col=0)
            word_to_vec = {word: np.array(eval(vec_str)) for word, vec_str in 
                          zip(df.index, df['vector'])}
            return word_to_vec
        except Exception as e:
            st.error(f"Error loading pre-computed embeddings: {e}")
            return generate_embeddings()
    else:
        return generate_embeddings()

@st.cache_data
def generate_embeddings():
    """Generate some pre-computed embeddings for common words using random vectors.
       In a real app, you would use pre-trained embeddings like GloVe or Word2Vec."""
    np.random.seed(42)  # For reproducibility
    
    # Common words for demonstration
    common_words = [
        # Gender & Royalty
        "king", "queen", "man", "woman", "prince", "princess", "boy", "girl", 
        "emperor", "empress", "duke", "duchess", "lord", "lady",
        
        # Countries & Capitals
        "france", "paris", "germany", "berlin", "japan", "tokyo", "italy", "rome",
        "spain", "madrid", "china", "beijing", "russia", "moscow", "brazil", "brasilia",
        "australia", "canberra", "canada", "ottawa",
        
        # Car brands with clear attributes
        "toyota", "honda", "ford", "volkswagen", "bmw", "mercedes", "audi", "porsche",
        "tesla", "jeep", "hybrid", "electric", "gasoline", "diesel", "compact", "midsize",
        "suv", "luxury", "sports",
        
        # Animals & Babies
        "dog", "puppy", "cat", "kitten", "horse", "foal", "cow", "calf",
        "sheep", "lamb", "goat", "kid", "lion", "cub", "tiger", "bear",
        "deer", "fawn", "kangaroo", "joey", "swan", "cygnet", "elephant", "calf",
        
        # Sports & Equipment
        "soccer", "basketball", "tennis", "golf", "hockey", "cricket", "rugby",
        "goal", "net", "racket", "club", "stick", "puck", "hoop", "bat"
    ]
    
    # For demonstration, we'll create "meaningful" embeddings for certain relationships
    word_to_vec = {}
    
    # Create base vectors (300-dimensional)
    dim = 300
    
    # Base vectors for different concepts
    male_vec = np.random.normal(0, 1, dim)
    female_vec = np.random.normal(0, 1, dim)
    royalty_vec = np.random.normal(0, 1, dim)
    country_vec = np.random.normal(0, 1, dim)
    capital_vec = np.random.normal(0, 1, dim)
    
    # Car related vectors
    car_brand_vec = np.random.normal(0, 1, dim)
    electric_vec = np.random.normal(0, 1, dim)
    gasoline_vec = np.random.normal(0, 1, dim)
    diesel_vec = np.random.normal(0, 1, dim)
    hybrid_vec = np.random.normal(0, 1, dim)
    size_vec = np.random.normal(0, 1, dim)
    luxury_vec = np.random.normal(0, 1, dim)
    sports_vec = np.random.normal(0, 1, dim)
    
    # Animal related vectors
    animal_vec = np.random.normal(0, 1, dim)
    baby_vec = np.random.normal(0, 1, dim)
    
    # Sports related vectors
    sport_vec = np.random.normal(0, 1, dim)
    equipment_vec = np.random.normal(0, 1, dim)
    
    # Create semantic relationships
    # Gender & Royalty - Enhanced relationships
    word_to_vec.update({
        "man": male_vec + 0.1 * np.random.normal(0, 1, dim),
        "woman": female_vec + 0.1 * np.random.normal(0, 1, dim),
        "king": male_vec + royalty_vec * 1.2,
        "queen": female_vec + royalty_vec * 1.2,
        "prince": male_vec + royalty_vec * 0.9,
        "princess": female_vec + royalty_vec * 0.9,
        "emperor": male_vec + royalty_vec * 1.5 + country_vec * 0.3,
        "empress": female_vec + royalty_vec * 1.5 + country_vec * 0.3,
        "duke": male_vec + royalty_vec * 0.7 + country_vec * 0.2,
        "duchess": female_vec + royalty_vec * 0.7 + country_vec * 0.2,
        "lord": male_vec + royalty_vec * 0.5,
        "lady": female_vec + royalty_vec * 0.5
    })
    
    # Countries & capitals with meaningful relationships
    for country, capital in [("france", "paris"), ("germany", "berlin"), ("japan", "tokyo"),
                            ("italy", "rome"), ("spain", "madrid"), ("china", "beijing"),
                            ("russia", "moscow"), ("brazil", "brasilia"), ("australia", "canberra"),
                            ("canada", "ottawa")]:
        country_vector = country_vec + np.random.normal(0, 0.1, dim)
        word_to_vec[country] = country_vector
        word_to_vec[capital] = capital_vec + 0.8 * country_vector
    
    # Car brands with specific attributes
    car_attributes = {
        "toyota": hybrid_vec * 0.7 + size_vec * 0.5,
        "honda": hybrid_vec * 0.6 + size_vec * 0.4,
        "ford": gasoline_vec * 0.8 + size_vec * 0.6,
        "volkswagen": diesel_vec * 0.7 + size_vec * 0.5,
        "bmw": luxury_vec * 1.0 + gasoline_vec * 0.6,
        "mercedes": luxury_vec * 1.1 + diesel_vec * 0.5,
        "audi": luxury_vec * 0.9 + gasoline_vec * 0.7,
        "porsche": luxury_vec * 1.2 + sports_vec * 0.8,
        "tesla": electric_vec * 1.5 + luxury_vec * 0.7,
        "jeep": diesel_vec * 0.9 + size_vec * 0.8
    }
    for brand, attributes in car_attributes.items():
        word_to_vec[brand] = car_brand_vec + attributes
    
    # Car features
    word_to_vec.update({
        "hybrid": hybrid_vec * 1.2,
        "electric": electric_vec * 1.3,
        "gasoline": gasoline_vec * 1.1,
        "diesel": diesel_vec * 1.1,
        "compact": size_vec * 0.3,
        "midsize": size_vec * 0.6,
        "suv": size_vec * 0.9,
        "luxury": luxury_vec * 1.0,
        "sports": luxury_vec * 0.8 + size_vec * 0.5
    })
    
    # Animals & Babies - Enhanced relationships
    animals = {
        "dog": ("puppy", 0.9),
        "cat": ("kitten", 0.85),
        "horse": ("foal", 0.88),
        "cow": ("calf", 0.92),
        "sheep": ("lamb", 0.87),
        "goat": ("kid", 0.84),
        "lion": ("cub", 0.95),
        "bear": ("cub", 0.93),
        "deer": ("fawn", 0.89),
        "kangaroo": ("joey", 0.91),
        "swan": ("cygnet", 0.86),
        "elephant": ("calf", 0.90)
    }
    for animal, (baby, strength) in animals.items():
        animal_vector = animal_vec + np.random.normal(0, 0.1, dim)
        word_to_vec[animal] = animal_vector
        word_to_vec[baby] = animal_vector * strength + baby_vec
    
    # Sports & Equipment - Updated with equipment focus
    sports = {
        "soccer": ("goal", "net"),
        "basketball": ("hoop",),
        "tennis": ("racket",),
        "golf": ("club",),
        "hockey": ("stick", "puck"),
        "cricket": ("bat",)
    }
    for sport, equipment in sports.items():
        sport_vector = sport_vec + np.random.normal(0, 0.1, dim)
        word_to_vec[sport] = sport_vector
        for item in equipment:
            word_to_vec[item] = equipment_vec + 0.8 * sport_vector
    
    # Fill in remaining words with random vectors
    for word in common_words:
        if word not in word_to_vec:
            word_to_vec[word] = np.random.normal(0, 1, dim)
    
    # Normalize the vectors
    for word in word_to_vec:
        word_to_vec[word] = word_to_vec[word] / np.linalg.norm(word_to_vec[word])
    
    # Save to CSV for future use
    df = pd.DataFrame({
        'vector': [str(list(vec)) for vec in word_to_vec.values()]
    }, index=list(word_to_vec.keys()))
    
    df.to_csv('glove_embeddings.csv')
    
    return word_to_vec

# Load embeddings
word_to_vec = load_embeddings()

# Sidebar - Word input section
with st.sidebar:
    st.header("Choose Words to Visualise")
    
    # Default word sets for different demonstrations
    word_sets = {
        "Gender & Royalty": ["king", "queen", "emperor", "empress", "prince", "princess", "duke", "duchess"],
        "Countries & Capitals": ["france", "paris", "germany", "berlin", "japan", "tokyo", "italy", "rome"],
        "Car Brands & Features": ["toyota", "honda", "tesla", "bmw", "hybrid", "electric", "gasoline", "luxury"],
        "Animals & Babies": ["dog", "puppy", "cat", "kitten", "lion", "cub", "elephant", "calf"],
        "Sports & Equipment": ["soccer", "goal", "basketball", "hoop", "tennis", "racket", "hockey", "stick"]
    }
    
    selected_set = st.selectbox("Choose a word set", options=list(word_sets.keys()))
    
    default_text = "\n".join(word_sets[selected_set])
    
    custom_words = st.text_area("Enter words (one per line)", default_text)
    words = [word.strip().lower() for word in custom_words.split("\n") if word.strip()]
    
    # Dimensionality reduction technique
    st.header("Visualisation Settings")
    dim_reduction = st.selectbox(
        "Dimensionality reduction technique",
        options=["PCA"],
        help="PCA captures the main variance in the data"
    )

# Get word vectors for visualization
word_vectors = []
valid_words = []
invalid_words = []

for word in words:
    if word in word_to_vec:
        word_vectors.append(word_to_vec[word])
        valid_words.append(word)
    else:
        invalid_words.append(word)

# Main visualization area
if len(word_vectors) < 3:
    st.error("Please enter at least 3 valid words to create a 3D visualisation.")
else:
    # Feedback on invalid words if any
    if invalid_words:
        st.warning(f"No vectors found for: {', '.join(invalid_words)}")
    
    # Progress indicator for dimensionality reduction
    with st.spinner(f"Applying {dim_reduction} to reduce dimensions to 3D..."):
        # Apply dimensionality reduction to get 3D coordinates
        if dim_reduction == "PCA":
            reducer = PCA(n_components=3)
            vectors_3d = reducer.fit_transform(word_vectors)
            explained_var = reducer.explained_variance_ratio_.sum()
            method_info = f"PCA (capturing {explained_var:.1%} of variance)"
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["3D Visualisation", "Word Relationships"])
    
    with tab1:
        st.subheader(f"Word Embeddings in 3D Space ({method_info})")
        
        # Create 3D visualization
        fig = go.Figure(data=[go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=list(range(len(valid_words))),
                colorscale='Viridis',
                opacity=0.8,
                symbol='circle'
            ),
            text=valid_words,
            hoverinfo='text',
            textposition="top center"
        )])
        
        # Update layout for better visualization
        fig.update_layout(
            height=600,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Display the 3D plot with a unique key
        st.plotly_chart(fig, use_container_width=True, key="main_chart")
    
    with tab2:
        st.subheader("Exploring Word Relationships")
        
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Word Analogies")
            st.markdown("Explore the classic word analogy task: **a is to b as c is to ?**")
            
            # Select words for analogy
            word1 = st.selectbox("Select word a", valid_words, index=0)
            word2 = st.selectbox("Select word b", valid_words, index=min(1, len(valid_words)-1))
            word3 = st.selectbox("Select word c", valid_words, index=min(2, len(valid_words)-1))
            
            if st.button("Calculate Analogy"):
                # Get indices of selected words for visualization
                idx1 = valid_words.index(word1)
                idx2 = valid_words.index(word2)
                idx3 = valid_words.index(word3)
                
                # Create a new figure for the analogy visualization
                analogy_fig = go.Figure(data=[go.Scatter3d(
                    x=vectors_3d[:, 0],
                    y=vectors_3d[:, 1],
                    z=vectors_3d[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=6,
                        color='lightgrey',
                        opacity=0.3
                    ),
                    text=valid_words,
                    hoverinfo='text'
                )])
                
                # Highlight the selected words
                for i, (idx, color) in enumerate([(idx1, 'blue'), (idx2, 'red'), (idx3, 'green')]):
                    analogy_fig.add_trace(go.Scatter3d(
                        x=[vectors_3d[idx, 0]],
                        y=[vectors_3d[idx, 1]],
                        z=[vectors_3d[idx, 2]],
                        mode='markers+text',
                        marker=dict(size=12, color=color),
                        text=[valid_words[idx]],
                        textposition="top center",
                        name=valid_words[idx]
                    ))
                
                # Draw lines between the words to show relationships
                analogy_fig.add_trace(go.Scatter3d(
                    x=[vectors_3d[idx1, 0], vectors_3d[idx2, 0]],
                    y=[vectors_3d[idx1, 1], vectors_3d[idx2, 1]],
                    z=[vectors_3d[idx1, 2], vectors_3d[idx2, 2]],
                    mode='lines',
                    line=dict(color='blue', width=5, dash='solid'),
                    name=f"{word1} → {word2}"
                ))
                
                analogy_fig.add_trace(go.Scatter3d(
                    x=[vectors_3d[idx1, 0], vectors_3d[idx3, 0]],
                    y=[vectors_3d[idx1, 1], vectors_3d[idx3, 1]],
                    z=[vectors_3d[idx1, 2], vectors_3d[idx3, 2]],
                    mode='lines',
                    line=dict(color='green', width=5, dash='dot'),
                    name=f"{word1} → {word3}"
                ))
                
                # Apply the same layout settings
                analogy_fig.update_layout(
                    height=500,
                    scene=dict(
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        zaxis_title='Dimension 3',
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                # Use a unique key for the analogy chart
                st.plotly_chart(analogy_fig, use_container_width=True, key="analogy_chart")
                
                # Compute and show analogies
                with st.spinner("Computing word analogies..."):
                    w1_vec = word_to_vec[word1]
                    w2_vec = word_to_vec[word2]
                    w3_vec = word_to_vec[word3]
                    
                    # Calculate: word2 - word1 + word3 should approximate word4
                    result_vec = w2_vec - w1_vec + w3_vec
                    
                    # Find most similar words to the result vector
                    most_similar = []
                    for word in word_to_vec:
                        if word.lower() not in [w.lower() for w in [word1, word2, word3]]:
                            vec = word_to_vec[word]
                            similarity = np.dot(vec, result_vec) / (np.linalg.norm(vec) * np.linalg.norm(result_vec))
                            most_similar.append((word, similarity))
                    
                    most_similar.sort(key=lambda x: x[1], reverse=True)
                    
                    st.success(f"Based on the word vectors, if **{word2}** is to **{word1}** as **X** is to **{word3}**, then **X** might be:")
                    
                    # Display the top results in a more visually appealing way
                    result_cols = st.columns(5)
                    for i, (word, sim) in enumerate(most_similar[:5]):
                        with result_cols[i]:
                            st.metric(
                                label=f"Result {i+1}",
                                value=word,
                                delta=f"Similarity: {sim:.3f}"
                            )
        
        with col2:
            st.markdown("### Cluster Analysis")
            st.markdown("See which words are closest to a selected word in the embedding space")
            
            # Select a word to analyze
            target_word = st.selectbox("Select a word to analyze", valid_words)
            
            if st.button("Find Similar Words"):
                target_idx = valid_words.index(target_word)
                target_vector = word_vectors[target_idx]
                
                # Calculate distances to all other words
                distances = []
                for i, word in enumerate(valid_words):
                    if i != target_idx:
                        vec = word_vectors[i]
                        # Cosine similarity
                        similarity = np.dot(target_vector, vec) / (np.linalg.norm(target_vector) * np.linalg.norm(vec))
                        distances.append((word, similarity))
                
                # Sort by similarity (highest first)
                distances.sort(key=lambda x: x[1], reverse=True)
                
                st.success(f"Words most similar to '**{target_word}**' in this set:")
                
                # Create a horizontal bar chart of similarities
                sim_words = [d[0] for d in distances[:8]]
                sim_values = [d[1] for d in distances[:8]]
                
                similarity_fig = go.Figure(go.Bar(
                    x=sim_values,
                    y=sim_words,
                    orientation='h',
                    marker_color='darkblue',
                    text=[f"{s:.3f}" for s in sim_values],
                    textposition='auto'
                ))
                
                similarity_fig.update_layout(
                    title=f"Similarity to '{target_word}'",
                    xaxis_title="Cosine Similarity",
                    yaxis_title="Word",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Use a unique key for the similarity chart
                st.plotly_chart(similarity_fig, use_container_width=True, key="similarity_chart")
                
                # Also show the words in the 3D space with the target highlighted
                highlight_fig = go.Figure(data=[go.Scatter3d(
                    x=vectors_3d[:, 0],
                    y=vectors_3d[:, 1],
                    z=vectors_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='lightgrey',
                        opacity=0.3
                    ),
                    text=valid_words,
                    hoverinfo='text'
                )])
                
                # Highlight the target word
                highlight_fig.add_trace(go.Scatter3d(
                    x=[vectors_3d[target_idx, 0]],
                    y=[vectors_3d[target_idx, 1]],
                    z=[vectors_3d[target_idx, 2]],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=[target_word],
                    textposition="top center",
                    name=target_word
                ))
                
                # Highlight the most similar words
                for word, sim in distances[:3]:
                    idx = valid_words.index(word)
                    highlight_fig.add_trace(go.Scatter3d(
                        x=[vectors_3d[idx, 0]],
                        y=[vectors_3d[idx, 1]],
                        z=[vectors_3d[idx, 2]],
                        mode='markers+text',
                        marker=dict(size=10, color='blue'),
                        text=[word],
                        textposition="top center",
                        name=word
                    ))
                    
                    # Draw a line connecting to the target word
                    highlight_fig.add_trace(go.Scatter3d(
                        x=[vectors_3d[target_idx, 0], vectors_3d[idx, 0]],
                        y=[vectors_3d[target_idx, 1], vectors_3d[idx, 1]],
                        z=[vectors_3d[target_idx, 2], vectors_3d[idx, 2]],
                        mode='lines',
                        line=dict(color='blue', width=3, dash='dot'),
                        name=f"{target_word} → {word}"
                    ))
                
                highlight_fig.update_layout(
                    height=500,
                    scene=dict(
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        zaxis_title='Dimension 3',
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                # Use a unique key for the highlight chart
                st.plotly_chart(highlight_fig, use_container_width=True, key="highlight_chart")

    # Educational section
    with st.expander("How to interpret this visualisation"):
        st.markdown("""
        ### Understanding Word Embeddings in 3D
        
        This visualisation shows how language models understand relationships between words by representing them in the vector space of a language model.
        
        - **Words that are close together** in this 3D space have similar meanings or are used in similar contexts
        - **The coloured points** represent individual words
        - **The lines** show relationships between words
        
        You can:
        - **Rotate** the visualisation by clicking and dragging
        - **Zoom** with your scroll wheel
        - **Pan** by right-clicking and dragging
        
        This demonstrates a key concept in how language models understand words - through their position in a multi-dimensional space!
        """)
        
        st.markdown("""
        ### The Math Behind Word Relationships
        
        When you see the classic example "king - man + woman = queen", it's showing that:
        
        1. The vector distance from 'man' to 'king' (representing the concept of 'royalty')
        2. When applied to 'woman' 
        3. Results in 'queen'
        
        This reveals how language models capture semantic relationships between words. In reality, these vectors have hundreds of dimensions, not just 3. The dimensionality reduction techniques (PCA or t-SNE) help us visualise these relationships in 3D space.
        """)
        
        # Add a conceptual diagram
        st.markdown("""
        ### Conceptual Representation
        
        #### What an LLM 'sees' in vector space:
        
        ```
        man + (king - man) ≈ king
        woman + (king - man) ≈ queen
        ```
        
        This is why LLMs can make analogies and understand relationships!
        """)
