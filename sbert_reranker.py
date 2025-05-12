from sentence_transformers import SentenceTransformer, util
import torch
import os

class SBERTReRanker:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_path="output/sbert_embeddings.pt"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.doc_id_to_index = {}
        self.cache_path = cache_path

    def encode_documents(self, raw_doc_texts, docIDs):
        if os.path.exists(self.cache_path):
            print("[SBERT] Loading cached embeddings...")
            cache = torch.load(self.cache_path)
            self.doc_embeddings = cache["embeddings"]
            self.doc_id_to_index = cache["id_to_index"]
        else:
            print("[SBERT] Computing new embeddings...")
            self.doc_embeddings = self.model.encode(raw_doc_texts, convert_to_tensor=True)
            self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(docIDs)}
            torch.save({
                "embeddings": self.doc_embeddings,
                "id_to_index": self.doc_id_to_index
            }, self.cache_path)
            print(f"[SBERT] Saved embeddings to {self.cache_path}")

    def rerank(self, query_str, candidate_doc_ids):
        """
        Rerank using cached embeddings
        """
        query_embedding = self.model.encode(query_str, convert_to_tensor=True)

        # Get only the embeddings for the candidate docIDs
        indices = [self.doc_id_to_index[doc_id] for doc_id in candidate_doc_ids]
        selected_doc_embeds = self.doc_embeddings[indices]

        # Cosine similarity
        cosine_scores = util.cos_sim(query_embedding, selected_doc_embeds)[0]
        ranked_indices = torch.argsort(cosine_scores, descending=True)

        reranked_docIDs = [candidate_doc_ids[i] for i in ranked_indices]
        return reranked_docIDs
