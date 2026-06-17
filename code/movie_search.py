# code/movie_search.py — embeddings + semantic search
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, runs locally

# Plot description is what we will embed
# Year/genre/actors/duration ride along as metadata, not embedded.
movies = [
    {
        "title": "Inception", "year": 2010, "genre": "Sci-Fi",
        "actors": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "duration_min": 148,
        "description": "A thief who steals secrets from people's subconscious is offered a chance to have his past crimes forgiven if he plants an idea into a target's mind.",
    },
    {
        "title": "The Shawshank Redemption", "year": 1994, "genre": "Drama",
        "actors": ["Tim Robbins", "Morgan Freeman"],
        "duration_min": 142,
        "description": "A banker is sentenced to life in prison for a murder he didn't commit and forms an unlikely friendship while planning an escape.",
    },
    {
        "title": "Mad Max: Fury Road", "year": 2015, "genre": "Action",
        "actors": ["Tom Hardy", "Charlize Theron"],
        "duration_min": 120,
        "description": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search of her homeland.",
    },
    {
        "title": "The Grand Budapest Hotel", "year": 2014, "genre": "Comedy",
        "actors": ["Ralph Fiennes", "Saoirse Ronan", "Tony Revolori"],
        "duration_min": 99,
        "description": "A concierge at a famous European hotel and his protege become friends and get tangled up in a theft and inheritance.",
    },
    {
        "title": "Get Out", "year": 2017, "genre": "Horror",
        "actors": ["Daniel Kaluuya", "Allison Williams"],
        "duration_min": 104,
        "description": "A young man visits his girlfriend's family estate and uncovers a disturbing secret.",
    },
    {
        "title": "Interstellar", "year": 2014, "genre": "Sci-Fi",
        "actors": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "duration_min": 169,
        "description": "A team of explorers travel through a wormhole in space in an attempt to save humanity.",
    },
    {
        "title": "La La Land", "year": 2016, "genre": "Musical/Romance",
        "actors": ["Ryan Gosling", "Emma Stone"],
        "duration_min": 128,
        "description": "A jazz pianist and an aspiring actress fall in love while pursuing their dreams in Los Angeles.",
    },
    {
        "title": "Parasite", "year": 2019, "genre": "Thriller/Drama",
        "actors": ["Song Kang-ho", "Lee Sun-kyun"],
        "duration_min": 132,
        "description": "A poor family schemes to become employed by a wealthy family, blurring the lines of class warfare.",
    },
]

descriptions = [m["description"] for m in movies]
embeddings = model.encode(descriptions)


def cosine_similarity(a, b):
    """
    Measures how similar two vectors' directions are, ignoring their length.
    np.dot(a, b) is the raw dot product — it grows with both how aligned
    the vectors are AND how long they are. Dividing by the product of their
    norms (lengths) cancels out the length effect, leaving a pure measure
    of a direction: 1.0 = pointing the same way (same meaning), 0 = unrelated,
    -1.0 = opposite. This is why two snippets with very different wording
    can still score high — embedding magnitude doesn't matter, only direction.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, top_k=8):
    # Embed the query with the SAME model used on the movie descriptions.
    # This step only makes sense because both live in the same vector
    # space — comparing vectors from two different models would be meaningless.
    query_vec = model.encode(query)

    # Brute-force comparison: check the query against every single movie
    # embedding, one by one. Fine for 8 movies; this is exactly the part
    # that doesn't scale, which is why we would need a real vector store
    # (Chroma) instead of a Python list + a for-loop.
    scores = [cosine_similarity(query_vec, emb) for emb in embeddings]

    # Pair each movie's metadata with its similarity score, then sort so
    # the highest-scoring (most similar) movie comes first.
    ranked = sorted(zip(movies, scores), key=lambda x: x[1], reverse=True)

    print(f"\nQuery: {query!r}")

    for movie, score in ranked[:top_k]:
        print(f"  {score:.3f}  {movie['title']} ({movie['year']}, {movie['genre']})")


if __name__ == "__main__":
    query = input("Search query: ")
    search(query)