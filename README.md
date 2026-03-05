
# 🎬 Hybrid Movie Recommendation System 🍿

[Live Demo](https://huggingface.co/spaces/meena1904/Movie_Recommendation_System)

---

## 🚀 Project Overview

A full-stack AI/ML web application for personalized movie recommendations:

* Users can search for movies or select their favorites.
* The system recommends movies using a **hybrid approach** (content-based + collaborative filtering).
* Provides an interactive and responsive interface for a smooth user experience.

---

## 💡 Features

* **Content-based Filtering**: Suggests movies similar to the selected ones using TF-IDF and metadata.
* **Collaborative Filtering**: Uses user ratings and preferences for personalized recommendations.
* **Hybrid Recommendation**: Combines both methods for more accurate suggestions.
* **Search Functionality**: Users can search movies by title or genre.
* **Responsive UI**: Built with Streamlit for instant deployment and interactivity.
* **Deployment-ready**: Hosted on Hugging Face Spaces.

---

## 🛠️ Tech Stack

* **Backend & ML**: Python, Pandas, NumPy, SciPy, Scikit-learn
* **ML Models**: TF-IDF vectorizer, Cosine Similarity, Collaborative filtering (matrix factorization / KNN)
* **Frontend**: Streamlit for web interface
* **Deployment**: Hugging Face Spaces
* **Data**: Movies dataset with user ratings (CSV / sparse matrices)

---

## 📂 Project Structure

```
hybrid-movie-recommender/
├── app.py                # Streamlit app
├── movies.csv            # Movie metadata
├── filtered_data.csv     # Preprocessed movie data
├── tfidf_matrix.npz      # Content-based TF-IDF features
├── tfidf_vectorizer.pkl  # Saved vectorizer
├── predicted_ratings.npz # Collaborative filtering predictions
├── requirements.txt
└── README.md
```

---

## 🖥️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/meena1904/Movie_Recommendation_System.git
cd Movie_Recommendation_System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

# 4. Open in your browser
# http://localhost:8501
```

---

## 📌 Notes

* **Hybrid Approach**: Content-based handles new movies, collaborative handles user preferences.
* **Sparse Matrix**: Predicted ratings stored as `.npz` to save memory and speed up computation.
* **Streamlit**: No need for Flask; frontend and backend integrated for easy deployment.
* **Deployment**: Live demo hosted on Hugging Face Spaces with all dependencies included.

---

## 🤝 Connect with Me

* [LinkedIn](https://linkedin.com/in/s-meenakshi-b2356b288)

---

