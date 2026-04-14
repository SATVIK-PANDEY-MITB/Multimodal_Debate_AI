# rag.py

from sentence_transformers import SentenceTransformer


import faiss

import numpy as np

import os
import re



# Load documents from docs/

docs_folder = "docs/"

documents = []

for filename in os.listdir(docs_folder):

    with open(os.path.join(docs_folder,filename),"r") as f:

        documents.append(f.read())

# Embedding model

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = embed_model.encode(documents)


# FAISS index for fast retrieval


dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(doc_embeddings))


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with"
}


def _extract_keywords(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def retrieve_context(query,k=1):

    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    results = [documents[i] for i in I[0]]

    

    query_keywords = set(_extract_keywords(query))
    filtered = []

    for text in results:
        text_words = set(_extract_keywords(text))
        if query_keywords and query_keywords.intersection(text_words):
            filtered.append(text)

    if not filtered:
        # Fall back to the nearest semantic result instead of returning empty context.
        filtered = results[:1]

    return "\n".join(filtered)
