# ==========================================
# Import Libraries
# ==========================================

import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("🎬 Movie Recommendation")

st.sidebar.markdown("---")

st.sidebar.header("About Project")

st.sidebar.write("""
This project recommends movies using

✔ TF-IDF Vectorizer

✔ Cosine Similarity

Dataset:
TMDB 5000 Movie Dataset
""")

st.sidebar.markdown("---")


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():

    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    movies = movies.merge(
        credits,
        on="title"
    )

    return movies


movies = load_data()


# ==========================================
# Keep Required Columns
# ==========================================

movies = movies[
    [
        "title",
        "overview",
        "genres",
        "keywords",
        "release_date",
        "runtime",
        "vote_average",
        "popularity",
        "original_language"
    ]
]


# ==========================================
# Remove Missing Values
# ==========================================

movies.dropna(inplace=True)


# ==========================================
# Convert Genres
# ==========================================

def convert(text):

    result = []

    for item in ast.literal_eval(text):

        result.append(item["name"])

    return result


movies["genres"] = movies["genres"].apply(convert)

movies["keywords"] = movies["keywords"].apply(convert)


# ==========================================
# Clean Genre & Keyword Names
# ==========================================

movies["genres"] = movies["genres"].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies["keywords"] = movies["keywords"].apply(
    lambda x: [i.replace(" ", "") for i in x]
)


# ==========================================
# Process Overview
# ==========================================

movies["overview"] = movies["overview"].apply(
    lambda x: x.split()
)


# ==========================================
# Create Tags
# ==========================================

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
)


movies["tags"] = movies["tags"].apply(
    lambda x: " ".join(x).lower()
)


# ==========================================
# TF-IDF Vectorizer
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english"
)

vectors = tfidf.fit_transform(
    movies["tags"]
)


# ==========================================
# Cosine Similarity
# ==========================================

similarity = cosine_similarity(vectors)


# ==========================================
# Sidebar Statistics
# ==========================================

st.sidebar.header("📊 Dataset Statistics")

st.sidebar.write(
    f"🎥 Total Movies : {len(movies)}"
)

st.sidebar.write(
    f"⭐ Average Rating : {movies['vote_average'].mean():.2f}"
)

st.sidebar.write(
    f"⏱ Average Runtime : {int(movies['runtime'].mean())} min"
)

st.sidebar.write(
    f"🌍 Languages : {movies['original_language'].nunique()}"
)

st.sidebar.write(
    f"🎭 Genres : {movies['genres'].explode().nunique()}"
)

st.sidebar.markdown("---")


# ==========================================
# Recommendation Settings
# ==========================================

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# ==========================================
# Main Title
# ==========================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Find similar movies using TF-IDF Vectorizer and Cosine Similarity."
)

st.divider()

# ==========================================
# Movie Search
# ==========================================

movie_list = sorted(movies["title"].unique())

selected_movie = st.selectbox(
    "🔍 Search Movie",
    movie_list,
    index=None,
    placeholder="Type or select a movie..."
)


# ==========================================
# Recommendation Function
# ==========================================

def recommend(movie_name, num_movies):

    # Get selected movie index
    movie_index = movies[movies["title"] == movie_name].index[0]

    # Similarity scores
    distances = similarity[movie_index]

    # Sort by similarity
    movie_distance = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []

    for i in movie_distance[1:num_movies + 1]:

        recommended_movies.append(
            movies.iloc[i[0]]
        )

    return recommended_movies


# ==========================================
# Display Movie Details
# ==========================================

if selected_movie:

    movie = movies[movies["title"] == selected_movie].iloc[0]

    st.subheader("🎥 Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write("###", movie["title"])

        st.write("⭐ Rating :", round(movie["vote_average"], 1))

        st.write("📅 Release :", movie["release_date"])

        st.write("🌍 Language :", movie["original_language"])

    with col2:

        st.write("⏱ Runtime :", int(movie["runtime"]), "Minutes")

        st.write("🔥 Popularity :", round(movie["popularity"], 2))

        genres = ", ".join(movie["genres"])

        st.write("🎭 Genres :", genres)

    st.subheader("📝 Overview")

    st.write(movie["overview"])

    st.divider()

    # ==========================================
# Recommend Movies
# ==========================================

if selected_movie:

    if st.button("🎬 Recommend Movies"):

        recommendations = recommend(
            selected_movie,
            num_recommendations
        )

        st.subheader("🎬 Recommended Movies")

        for movie in recommendations:

            with st.container():

                st.markdown(f"## 🎥 {movie['title']}")

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "⭐ Rating :",
                        round(movie["vote_average"], 1)
                    )

                    st.write(
                        "📅 Release :",
                        movie["release_date"]
                    )

                    st.write(
                        "🌍 Language :",
                        movie["original_language"]
                    )

                with col2:

                    st.write(
                        "⏱ Runtime :",
                        int(movie["runtime"]),
                        "Minutes"
                    )

                    st.write(
                        "🔥 Popularity :",
                        round(movie["popularity"], 2)
                    )

                    genres = ", ".join(movie["genres"])

                    st.write(
                        "🎭 Genres :",
                        genres
                    )

                st.divider()


# ==========================================
# Dashboard
# ==========================================

st.header("📊 Movie Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎬 Movies", len(movies))

with col2:
    st.metric(
        "⭐ Avg Rating",
        round(movies["vote_average"].mean(), 2)
    )

with col3:
    st.metric(
        "⏱ Avg Runtime",
        str(int(movies["runtime"].mean())) + " min"
    )

with col4:
    st.metric(
        "🌍 Languages",
        movies["original_language"].nunique()
    )

st.divider()

# ==========================================
# Genre Filter
# ==========================================

all_genres = sorted(
    set(
        genre
        for genres in movies["genres"]
        for genre in genres
    )
)

selected_genre = st.selectbox(
    "🎭 Filter by Genre",
    ["All"] + all_genres
)

if selected_genre != "All":

    filtered_movies = movies[
        movies["genres"].apply(
            lambda x: selected_genre in x
        )
    ]

else:

    filtered_movies = movies.copy()

# ==========================================
# Language Filter
# ==========================================

languages = sorted(
    filtered_movies["original_language"].unique()
)

selected_language = st.selectbox(
    "🌍 Filter by Language",
    ["All"] + list(languages)
)

if selected_language != "All":

    filtered_movies = filtered_movies[
        filtered_movies["original_language"] == selected_language
    ]

# ==========================================
# Rating Filter
# ==========================================

rating = st.slider(
    "⭐ Minimum Rating",
    0.0,
    10.0,
    5.0
)

filtered_movies = filtered_movies[
    filtered_movies["vote_average"] >= rating
]

# ==========================================
# Top Rated Movies
# ==========================================

st.subheader("⭐ Top Rated Movies")

top_movies = filtered_movies.sort_values(
    by="vote_average",
    ascending=False
).head(10)

st.dataframe(
    top_movies[
        [
            "title",
            "vote_average",
            "runtime",
            "release_date"
        ]
    ],
    use_container_width=True
)
# ==========================================
# Most Popular Movies
# ==========================================

st.subheader("🔥 Most Popular Movies")

popular_movies = filtered_movies.sort_values(
    by="popularity",
    ascending=False
).head(10)

st.dataframe(
    popular_movies[
        [
            "title",
            "popularity",
            "vote_average"
        ]
    ],
    use_container_width=True
)


st.markdown(
    """
    <center>
    <h4>🎬 Movie Recommendation System</h4>
    <p>Developed using Streamlit, TF-IDF and Cosine Similarity</p>
    </center>
    """,
    unsafe_allow_html=True
)
