import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Movie reviews dataset
movie_reviews = [
    "This movie was fantastic! The storyline was gripping, and the acting was top-notch.",
    "I didn't enjoy the movie. It was too slow and predictable.",
    "An absolute masterpiece! The visuals and soundtrack were breathtaking.",
    "The movie was average. Some scenes were good, but others felt unnecessary.",
    "Worst movie I've seen this year. Poor acting and a weak plot."
]

# Sentence Tokenization
print("\nStep 4: Sentence Tokenization\n")
all_sentences = []
for review in movie_reviews:
    doc = nlp(review)
    sentences = [sent.text for sent in doc.sents]
    all_sentences.extend(sentences)
    print(f"Review: {review}")
    print(f"Sentences: {sentences}\n")

# Feature Extraction - Bag of Words (BoW)
print("\nStep 5: Feature Extraction - Bag of Words (BoW)\n")
vectorizer_bow = CountVectorizer()
bow_matrix = vectorizer_bow.fit_transform(movie_reviews)
print("Vocabulary (BoW):")
print(vectorizer_bow.get_feature_names_out())
print("\nBag of Words Matrix:")
print(pd.DataFrame(bow_matrix.toarray(), columns=vectorizer_bow.get_feature_names_out()))
