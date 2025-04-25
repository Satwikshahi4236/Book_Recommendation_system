
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    'BookID': [1, 2, 3, 4],
    'Title': ['Book A', 'Book B', 'Book C', 'Book D'],
    'Genre': ['Fantasy', 'Science Fiction', 'Fantasy', 'Mystery']
}
books = pd.DataFrame(data)

user_interested_genres = ['Fantasy', 'Mystery']


count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(books['Genre'])


similarity = cosine_similarity(genre_matrix)


def recommend_books(user_genres, books, similarity):
    recommendations = []
    for genre in user_genres:
        genre_index = books.index[books['Genre'] == genre].tolist()
        if genre_index:
            sim_scores = similarity[genre_index]
            sorted_indices = sim_scores.argsort()[::-1].flatten()
            for idx in sorted_indices:
                recommendations.append(books.iloc[idx]['Title'])
    return recommendations

recommendations = recommend_books(user_interested_genres, books, similarity)
print("Recommended books:", recommendations)