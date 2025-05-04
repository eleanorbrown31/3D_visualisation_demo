# 3D_visualisation_demo
# 3D Word Relationships Visualiser for LLMs

This Streamlit application creates an interactive 3D visualisation showing how language models relate words to one another in vector space. It's designed for educational purposes to help people new to AI understand word embeddings and semantic relationships.

## Features

- **Interactive 3D visualisation** of word embeddings using Plotly
- **Word relationship analysis** showing how words connect to each other
- **Analogy demonstrations** (like the classic "king - man + woman = queen")
- **Cluster analysis** showing which words are semantically closer to others
- **Multiple dimensionality reduction techniques** (PCA and t-SNE)
- **Pre-configured word sets** for different types of demonstrations
- **Educational explanations** about word embeddings and vector space

## Demo

![App Screenshot](https://placeholder.com/screenshot.png)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/word-embedding-visualiser.git
   cd word-embedding-visualiser
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required spaCy models:
   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download en_core_web_lg
   ```

## Usage

Run the Streamlit app locally:
```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

## How It Works

This application leverages pre-trained word vectors from spaCy language models to visualise word relationships. Here's what happens:

1. The app loads word vectors for user-selected words
2. It applies dimensionality reduction (PCA or t-SNE) to project the high-dimensional vectors to 3D
3. The words are plotted as points in 3D space, where proximity indicates semantic similarity
4. Users can explore relationships between words through analogy calculations and nearest neighbor analysis

## Deployment

The app can be deployed to Streamlit Cloud directly from GitHub:

1. Fork this repository or push it to your GitHub account
2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Deploy the app by selecting your repository and the main file (app.py)

## Presentation Tips

When using this for a presentation:
- Start with the "Gender & Royalty" word set to demonstrate the classic king/queen relationship
- Show how words cluster by meaning by using the "Professions" or "Emotions" sets
- Demonstrate analogies like "king is to man as queen is to woman" to show how the model captures relationships
- Explain that this is a simplified 3D projection of what's actually happening in hundreds of dimensions

## Customisation

You can modify `app.py` to:
- Add more pre-configured word sets
- Change the visualisation styling
- Add additional analysis methods
- Integrate with other language models

## Requirements

- Python 3.7+
- Libraries listed in requirements.txt

## License

MIT License

## Acknowledgements

This application uses:
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for 3D visualisations
- [spaCy](https://spacy.io/) for word embeddings
- [scikit-learn](https://scikit-learn.org/) for dimensionality reduction
