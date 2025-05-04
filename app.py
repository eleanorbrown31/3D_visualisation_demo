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
        # Professions
        "doctor", "nurse", "engineer", "teacher", "programmer", "artist", "scientist", "lawyer",
        # Technology
        "computer", "internet", "software", "hardware", "algorithm", "data", "network", "server",
        # Emotions
        "happy", "sad", "angry", "excited", "calm", "anxious", "confused", "joyful",
        # Countries & Capitals
        "france", "paris", "germany", "berlin", "japan", "tokyo", "italy", "rome",
        # Animals
        "dog", "cat", "bird", "fish", "lion", "tiger", "elephant", "monkey",
        # Food
        "apple", "banana", "pizza", "pasta", "rice", "bread", "cheese", "meat",
        # Transport
        "car", "train", "airplane", "bicycle", "ship", "bus", "motorcycle", "truck",
        # Household
        "table", "chair", "bed", "sofa", "kitchen", "bathroom", "window", "door",
        # Nature
        "tree", "flower", "mountain", "river", "ocean", "forest", "desert", "sky"
    ]
    
    # For demonstration, we'll create "meaningful" embeddings for certain relationships
    word_to_vec = {}
    
    # Create base vectors (300-dimensional)
    dim = 300
    
    # Base vectors for male and female concepts
    male_vec = np.random.normal(0, 1, dim)
    female_vec = np.random.normal(0, 1, dim)
    
    # Base vectors for royalty, profession, etc.
    royalty_vec = np.random.normal(0, 1, dim)
    profession_vec = np.random.normal(0, 1, dim)
    tech_vec = np.random.normal(0, 1, dim)
    emotion_vec = np.random.normal(0, 1, dim)
    country_vec = np.random.normal(0, 1, dim)
    capital_vec = np.random.normal(0, 1, dim)
    
    # Create semantic relationships
    word_to_vec["man"] = male_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["woman"] = female_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["king"] = male_vec + royalty_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["queen"] = female_vec + royalty_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["prince"] = male_vec + 0.8 * royalty_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["princess"] = female_vec + 0.8 * royalty_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["boy"] = 0.7 * male_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["girl"] = 0.7 * female_vec + 0.1 * np.random.normal(0, 1, dim)
    
    # Professions with intentional biases (for demonstration)
    word_to_vec["doctor"] = profession_vec + 0.3 * male_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["nurse"] = profession_vec + 0.3 * female_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["engineer"] = profession_vec + 0.4 * male_vec + 0.4 * tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["teacher"] = profession_vec + 0.2 * female_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["programmer"] = profession_vec + 0.3 * male_vec + 0.5 * tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["artist"] = profession_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["scientist"] = profession_vec + 0.2 * male_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["lawyer"] = profession_vec + 0.1 * np.random.normal(0, 1, dim)
    
    # Technology words
    word_to_vec["computer"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["internet"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["software"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["hardware"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["algorithm"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["data"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["network"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["server"] = tech_vec + 0.1 * np.random.normal(0, 1, dim)
    
    # Emotions
    positive_emotion = np.random.normal(0, 1, dim)
    negative_emotion = np.random.normal(0, 1, dim)
    word_to_vec["happy"] = emotion_vec + positive_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["sad"] = emotion_vec + negative_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["angry"] = emotion_vec + negative_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["excited"] = emotion_vec + positive_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["calm"] = emotion_vec + 0.5 * positive_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["anxious"] = emotion_vec + negative_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["confused"] = emotion_vec + 0.5 * negative_emotion + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["joyful"] = emotion_vec + positive_emotion + 0.1 * np.random.normal(0, 1, dim)
    
    # Countries & capitals with meaningful relationships
    word_to_vec["france"] = country_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["paris"] = capital_vec + 0.8 * word_to_vec["france"] + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["germany"] = country_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["berlin"] = capital_vec + 0.8 * word_to_vec["germany"] + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["japan"] = country_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["tokyo"] = capital_vec + 0.8 * word_to_vec["japan"] + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["italy"] = country_vec + 0.1 * np.random.normal(0, 1, dim)
    word_to_vec["rome"] = capital_vec + 0.8 * word_to_vec["italy"] + 0.1 * np.random.normal(0, 1, dim)
    
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
        "Gender & Royalty": ["king", "queen", "man", "woman", "prince", "princess"],
        "Professions": ["doctor", "nurse", "engineer", "teacher", "programmer", "artist"],
        "Technology": ["computer", "internet", "software", "hardware", "algorithm", "data"],
        "Emotions": ["happy", "sad", "angry", "excited", "calm", "anxious"],
        "Countries & Capitals": ["france", "paris", "germany", "berlin", "japan", "tokyo"],
        "Custom": []
    }
    
    selected_set = st.selectbox("Choose a word set", options=list(word_sets.keys()))
    
    if selected_set == "Custom":
        default_text = "\n".join(word_sets["Gender & Royalty"])
    else:
        default_text = "\n".join(word_sets[selected_set])
    
    custom_words = st.text_area("Enter words (one per line)", default_text)
    words = [word.strip().lower() for word in custom_words.split("\n") if word.strip()]
    
    # Dimensionality reduction technique
    st.header("Visualisation Settings")
    dim_reduction = st.selectbox(
        "Dimensionality reduction technique",
        options=["PCA", "t-SNE"],
        help="PCA is faster but t-SNE often shows clusters better"
    )
    
    # Additional parameters for t-SNE
    if dim_reduction == "t-SNE":
        perplexity = st.slider("t-SNE perplexity", min_value=5, max_value=50, value=30,
                              help="Higher values consider more global structure")
        n_iter = st.slider("t-SNE iterations", min_value=250, max_value=2000, value=1000, 
                          step=250, help="More iterations may give better results but take longer")

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
        else:  # t-SNE
            start_time = time.time()
            reducer = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
            vectors_3d = reducer.fit_transform(word_vectors)
            time_taken = time.time() - start_time
            method_info = f"t-SNE (computed in {time_taken:.1f}s)"
    
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
        
        # Display the 3D plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interactive controls for visualization options
        cols = st.columns(3)
        with cols[0]:
            show_labels = st.checkbox("Show word labels", value=True)
        with cols[1]:
            marker_size = st.slider("Point size", 3, 15, 8)
        with cols[2]:
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.1)
        
        # Update visualization based on controls
        if not show_labels:
            fig.update_traces(text=None, mode='markers')
        fig.update_traces(marker=dict(size=marker_size, opacity=opacity))
        
        st.plotly_chart(fig, use_container_width=True)
    
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
                
                st.plotly_chart(analogy_fig, use_container_width=True)
                
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
                
                st.plotly_chart(similarity_fig, use_container_width=True)
                
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
                
                st.plotly_chart(highlight_fig, use_container_width=True)

    # Educational section
    with st.expander("How to interpret this visualisation"):
        st.markdown("""
        ### Understanding Word Embeddings in 3D
        
        This visualisation shows how words are related to each other in the vector space of a language model.
        
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

# Deployment instructions at the bottom
with st.expander("Deployment Instructions"):
    st.markdown("""
    ### How to Deploy This App on GitHub and Streamlit
    
    1. **Create a GitHub repository**:
       - Create a new repository on GitHub
       - Clone it to your local machine
    
    2. **Set up the project structure**:
       - Save this code as `app.py`
       - Create a `requirements.txt` file with these dependencies:
         ```
         streamlit>=1.22.0
         plotly>=5.14.0
         numpy>=1.24.0
         scikit-learn>=1.2.0
         pandas>=1.5.0
         ```
    
    3. **Push to GitHub**:
       - Add and commit your files
       - Push to your GitHub repository
    
    4. **Deploy on Streamlit Cloud**:
       - Go to [Streamlit Cloud](https://streamlit.io/cloud)
       - Sign in with your GitHub account
       - Choose your repository, branch, and the main file (`app.py`)
       - Deploy!
       
    Your app will be deployed with a public URL that you can share with others or embed in presentations.
    """)

# Add a footer with additional resources
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Created for an AI educational presentation | <a href="https://github.com/yourusername/your-repo">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)