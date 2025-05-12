from symspellpy.symspellpy import SymSpell, Verbosity
import os
import json

class SpellCorrector:
    def __init__(self, dict_path="data/frequency_dictionary_en_82_765.txt",
                 domain_cache_path="output/domain_terms.json", max_edit_distance=2):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self.cache_path = domain_cache_path

        # Load general dictionary
        dictionary_path = os.path.abspath(dict_path)
        if not self.sym_spell.load_dictionary(dictionary_path, 0, 1):
            raise FileNotFoundError("SymSpell dictionary not found at: " + dictionary_path)

        # Try to load cached domain terms
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                domain_terms = json.load(f)
            print(f"[SpellCorrector] Loaded {len(domain_terms)} domain terms from cache.")
            for token, freq in domain_terms.items():
                self.sym_spell.create_dictionary_entry(token, freq)
        else:
            print("[SpellCorrector] Domain term cache not found. Please build it via add_domain_terms().")

    def add_domain_terms(self, token_freq_dict):
        """
        Add and cache domain-specific terms from corpus.
        """
        valid_terms = {
            token: freq for token, freq in token_freq_dict.items()
            if token.isalpha() and len(token) > 2
        }

        for token, freq in valid_terms.items():
            self.sym_spell.create_dictionary_entry(token, freq)

        # Save to persistent cache
        with open(self.cache_path, 'w') as f:
            json.dump(valid_terms, f)
        print(f"[SpellCorrector] Cached {len(valid_terms)} domain terms to {self.cache_path}")

    def correct_token(self, token):
        suggestions = self.sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
        return suggestions[0].term if suggestions else token

    def correct_query(self, query_tokens):
        return [self.correct_token(tok) for tok in query_tokens]
