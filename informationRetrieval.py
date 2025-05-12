from util import *

# Add your import statements here
import os
from rank_bm25 import BM25Okapi
from sbert_reranker import SBERTReRanker
from collections import Counter
from spell_corrector import SpellCorrector
from sklearn.feature_extraction.text import TfidfVectorizer

class InformationRetrieval():
    def __init__(self, args):
        self.args = args
        self.bm25 = None
        self.vectorizer = None
        self.doc_vectors = None
        self.corpus_tokens = None
        self.docIDs = None
        self.original_docs = None
        if self.args.rerank:
            # domain_specific_embeddings = f"{self.args.out_folder}{self.args.dataset}_sbert_embeddings.pt"
            # self.sbert = SBERTReRanker(cache_path=domain_specific_embeddings)
            self.sbert = SBERTReRanker()
        if self.args.spell:
            # domain_specific_dict = f"{self.args.out_folder}{self.args.dataset}_domain_terms.json"
            # self.spell_corrector = SpellCorrector(domain_cache_path=domain_specific_dict)
            self.spell_corrector = SpellCorrector()
            
       

    def buildIndex(self, docs, docIDs, raw_docs=None):
        """
        Builds the document index using BM25 weighting scheme.

        Parameters
        ----------
        docs : list of list of list of str
            A list of documents, each a list of sentences, each a list of words.
        docIDs : list of int
            List of document IDs.
        """
        # print(f"[DEBUG] Received {len(raw_docs)} raw docs") if raw_docs else print("[DEBUG] raw_docs is None")
        self.docIDs = docIDs
        self.original_docs = raw_docs
        self.corpus_tokens = []

        corpus = []
        for doc in docs:
            flat_tokens = []
            for sent in doc:
                flat_tokens.extend(sent)
            self.corpus_tokens.append(flat_tokens)
            corpus.append(" ".join(flat_tokens))  # For TF-IDF

        if self.args.use_tfidf:
            print("[IR] Using TF-IDF for document scoring.")
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
        else:
            print("[IR] Using BM25 for document scoring.")
            self.bm25 = BM25Okapi(self.corpus_tokens)

        # SBERT embedding (unchanged)
        if self.args.rerank and raw_docs is not None:
            self.sbert.encode_documents(raw_docs, docIDs)

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query using BM25 and SBERT hybrid.

        Parameters
        ----------
        queries : list of list of list of str
            A list of queries, each a list of sentences, each a list of words.

        Returns
        -------
        list of list of int
            Ranked list of document IDs for each query.
        """
        doc_IDs_ordered = []

        for idx, query in enumerate(queries):
            query_tokens = []
            for sent in query:
                query_tokens.extend(sent)
            
            original_tokens = query_tokens.copy()  # Save before expansion
            
            # Only apply query expansion if enabled
            if self.args.expand and len(query_tokens) < 4:
                from query_expansion import expand_query_pos_filtered
                expanded_tokens = expand_query_pos_filtered(query_tokens)
            else:
                expanded_tokens = []
            query_tokens = original_tokens * 3 + expanded_tokens
            
            # print(f"\n[Query {idx+1}] Original: {' '.join(original_tokens)}")
            # print(f"[Query {idx+1}] Expanded: {' '.join(query_tokens)}")

            if self.args.use_tfidf:
                # TF-IDF scoring
                query_str = " ".join(query_tokens)
                query_vec = self.vectorizer.transform([query_str])
                scores = (self.doc_vectors @ query_vec.T).toarray().flatten()
            else:
                # BM25 scoring
                scores = self.bm25.get_scores(query_tokens)
            top_k = 50
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            # Save BM25 scores and doc info
            top_docIDs = [self.docIDs[i] for i in ranked_indices]
            top_docTexts = [self.original_docs[i] for i in ranked_indices]
            bm25_scores = [scores[i] for i in ranked_indices]

            # Flatten query for SBERT
            query_str = " ".join(query_tokens)


            # Only apply SBERT reranking if flag is set
            if self.args.rerank:
                reranked_ids = self.sbert.rerank(query_str, top_docIDs)
                id_to_sbert = {doc_id: 1 - i / len(reranked_ids) for i, doc_id in enumerate(reranked_ids)}
            else:
                reranked_ids = top_docIDs
                id_to_sbert = {doc_id: 1.0 for doc_id in top_docIDs}  # fallback


            # Combine BM25 and SBERT scores
            id_to_bm25 = dict(zip(top_docIDs, bm25_scores))
            id_to_sbert = {doc_id: 1 - i / len(reranked_ids) for i, doc_id in enumerate(reranked_ids)}

            combined = [
                (doc_id, 0.7 * id_to_bm25.get(doc_id, 0) + 0.3 * id_to_sbert.get(doc_id, 0))
                for doc_id in top_docIDs
            ]
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
            final_reranked_docIDs = [doc_id for doc_id, _ in combined_sorted]

            # Append for this query
            doc_IDs_ordered.append(final_reranked_docIDs)

        return doc_IDs_ordered
