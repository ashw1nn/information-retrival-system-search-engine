from symspellpy.symspellpy import SymSpell, Verbosity
import os, json
from collections import Counter

class SpellCorrector:
    def __init__(self, dict_path="metadata/frequency_dictionary_en_82_765.txt",
                 domain_cache_path="output/domain_terms.json", max_edit_distance=2):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self.cache_path = domain_cache_path

        dictionary_path = os.path.abspath(dict_path)
        if not self.sym_spell.load_dictionary(dictionary_path, 0, 1):
            raise FileNotFoundError("SymSpell dictionary not found at: " + dictionary_path)

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                domain_terms = json.load(f)
            print(f"[SpellCorrector] Loaded {len(domain_terms)} domain terms from cache.")
            for token, freq in domain_terms.items():
                self.sym_spell.create_dictionary_entry(token, freq)
        else:
            print("[SpellCorrector] Domain term cache not found. It will be built on first correction.")

    def _build_from_docs(self, docs):
        flat_tokens = [token.lower() for doc in docs for sent in doc for token in sent if token.isalpha()]
        freq_dict = dict(Counter(flat_tokens))
        valid_terms = {token: freq for token, freq in freq_dict.items() if len(token) > 2}
        for token, freq in valid_terms.items():
            self.sym_spell.create_dictionary_entry(token, freq)
        with open(self.cache_path, 'w') as f:
            json.dump(valid_terms, f)
        print(f"[SpellCorrector] Built and cached {len(valid_terms)} domain terms.")

    def correct_query(self, query_tokens):
        # Lazy domain term loading
        if not os.path.exists(self.cache_path):
            try:
                from informationRetrieval import get_tokenized_docs
                docs = get_tokenized_docs()
                self._build_from_docs(docs)
            except Exception as e:
                raise RuntimeError("SpellCorrector failed to auto-load domain terms. Ensure tokenized docs are accessible.") from e
        return [self.correct_token(tok) for tok in query_tokens]

    def correct_token(self, token):
        suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
        return suggestions[0].term if suggestions else token
