# models.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
import numpy as np

# ---- Helper ----
def flatten_docs(docs):
    return [" ".join([" ".join(sent) for sent in doc]) for doc in docs]


# ---- LSA ----
def run_lsa_model(docs, queries, doc_ids):
    flat_docs = flatten_docs(docs)
    flat_queries = flatten_docs(queries)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(flat_docs)
    svd = TruncatedSVD(n_components=50,random_state=42) 
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    doc_IDs_ordered = []
    for query in flat_queries:
        query_vec = svd.transform(tfidf.transform([query]))
        scores = cosine_similarity(query_vec, lsa_matrix).flatten()
        ranked_ids = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        doc_IDs_ordered.append(ranked_ids)
    return doc_IDs_ordered


# ---- SBERT ----
def run_sbert_model(docs, queries, doc_ids):
    flat_docs = flatten_docs(docs)
    flat_queries = flatten_docs(queries)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(flat_docs, convert_to_numpy=True)
    doc_IDs_ordered = []
    for query in flat_queries:
        query_vec = model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_vec, doc_embeddings).flatten()
        ranked_ids = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        doc_IDs_ordered.append(ranked_ids)
    return doc_IDs_ordered


# ---- PLSA ----
def run_plsa_model(docs, queries, doc_ids, num_topics=100):
    tokenized_docs = [[token for sent in doc for token in sent] for doc in docs]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    doc_topics = np.array([model.get_document_topics(bow, minimum_probability=0.0) for bow in corpus])
    doc_vectors = np.array([[prob for _, prob in doc] for doc in doc_topics])

    doc_IDs_ordered = []
    for query in queries:
        query_tokens = [token for sent in query for token in sent]
        bow = dictionary.doc2bow(query_tokens)
        query_topic = model.get_document_topics(bow, minimum_probability=0.0)
        query_vector = np.array([prob for _, prob in query_topic]).reshape(1, -1)

        scores = cosine_similarity(query_vector, doc_vectors).flatten()
        ranked_ids = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        doc_IDs_ordered.append(ranked_ids)
    return doc_IDs_ordered


# ---- Hybrid (LSA + SBERT) ----
def run_hybrid_model(docs, queries, doc_ids):
    flat_docs = flatten_docs(docs)
    flat_queries = flatten_docs(queries)

    # LSA
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(flat_docs)
    svd = TruncatedSVD(n_components=100, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    # SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_matrix = model.encode(flat_docs, convert_to_numpy=True)

    doc_IDs_ordered = []
    for q in flat_queries:
        tfidf_q = tfidf.transform([q])
        lsa_q = svd.transform(tfidf_q)
        sbert_q = model.encode([q], convert_to_numpy=True)

        # Combine scores
        lsa_scores = cosine_similarity(lsa_q, lsa_matrix).flatten()
        sbert_scores = cosine_similarity(sbert_q, sbert_matrix).flatten()
        scores = (lsa_scores + sbert_scores) / 2

        ranked_ids = [doc_ids[i] for i in np.argsort(scores)[::-1]]
        doc_IDs_ordered.append(ranked_ids)
    return doc_IDs_ordered
