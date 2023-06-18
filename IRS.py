import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os

# Define the documents

# Download stopwords from NLTK


# Preprocess documents and queries
def preprocess(text):
    # Convert to lowercase and split into tokens
    tokens = text.lower().split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

# Build the inverted index
def build_inverted_index(documents):
    inverted_index = {}

    for doc_id, doc in enumerate(documents):
        tokens = preprocess(doc)

        # Compute term frequencies
        tf = Counter(tokens)

        # Update inverted index for each token
        for token in tf:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((doc_id, tf[token]))

    return inverted_index

# Compute TF-IDF scores
def compute_tfidf(inverted_index, documents):
    num_docs = len(documents)
    tfidf_scores = {}

    for term, postings in inverted_index.items():
        idf = math.log(num_docs / len(postings))

        for doc_id, tf in postings:
            if doc_id not in tfidf_scores:
                tfidf_scores[doc_id] = {}
            tfidf_scores[doc_id][term] = tf * idf

    return tfidf_scores
def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) & set(vec2))
    magnitude_vec1 = math.sqrt(sum(value**2 for value in vec1.values()))
    magnitude_vec2 = math.sqrt(sum(value**2 for value in vec2.values()))

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0

    return dot_product / (magnitude_vec1 * magnitude_vec2)
# Perform a query using the vector space model
def query_vsm(query, inverted_index, tfidf_scores, documents):
    query_tokens = preprocess(query)
    query_vector = Counter(query_tokens)
    query_norm = math.sqrt(sum(tf**2 for tf in query_vector.values()))

    doc_scores = {}

    for term, query_tf in query_vector.items():
        if term not in inverted_index:
            continue

        idf = math.log(len(documents) / len(inverted_index[term]))

        for doc_id, doc_tfidf in tfidf_scores.items():
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0

            doc_scores[doc_id] += query_tf * doc_tfidf.get(term, 0) * idf

    # Sort the documents by score in descending order
    ranked_docs = [(doc_id, cosine_similarity(query_vector, doc_tfidf)) for doc_id, doc_tfidf in tfidf_scores.items()]
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_docs

def read_documents_from_folder(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                document = file.read()
                documents.append(document)



    return documents

# Get the folder path from the user
folder_path = "doc"

# Read documents from the folder
documents = read_documents_from_folder(folder_path)

# Build the inverted index
inverted_index = build_inverted_index(documents)

# Compute TF-IDF scores
tfidf_scores = compute_tfidf(inverted_index, documents)

# Accept queries from the user
while True:
    query = input("Enter a query (or press 'q' to quit): ")

    if query.lower() == 'q':
        break

    ranked_docs = query_vsm(query, inverted_index, tfidf_scores, documents)

    print("Ranked Documents:")
    if ranked_docs:
        for doc_id, score in ranked_docs:
            print(documents[doc_id], " - Score:", score)
    else:
        print("No matching documents")

    print()

