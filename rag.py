import os
import re


try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


try:
    import faiss
except Exception:
    faiss = None


# Load documents from docs/
docs_folder = "docs/"
documents = []

for filename in os.listdir(docs_folder):
    if not filename.lower().endswith(".txt"):
        continue
    with open(os.path.join(docs_folder, filename), "r", encoding="utf-8") as file_handle:
        documents.append(file_handle.read())


embed_model = None
doc_embeddings = None
index = None

if SentenceTransformer is not None and faiss is not None:
    try:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = embed_model.encode(documents)
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(doc_embeddings)
    except Exception:
        embed_model = None
        doc_embeddings = None
        index = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with"
}


def _extract_keywords(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _rank_documents_by_overlap(query):
    query_keywords = set(_extract_keywords(query))
    if not query_keywords:
        return documents[:1]

    ranked = []
    for text in documents:
        text_keywords = set(_extract_keywords(text))
        score = len(query_keywords.intersection(text_keywords))
        if score > 0:
            ranked.append((score, text))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in ranked]


def retrieve_context(query, k=1):
    if embed_model is not None and index is not None:
        query_vec = embed_model.encode([query])
        _, indices = index.search(query_vec, k)
        results = [documents[i] for i in indices[0]]
    else:
        results = _rank_documents_by_overlap(query)[:k]

    query_keywords = set(_extract_keywords(query))
    filtered = []

    for text in results:
        text_words = set(_extract_keywords(text))
        if query_keywords and query_keywords.intersection(text_words):
            filtered.append(text)

    if not filtered:
        filtered = results[:1]

    return "\n".join(filtered)
