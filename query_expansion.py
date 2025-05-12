import nltk
from nltk.corpus import wordnet
# nltk.download("averaged_perceptron_tagger")

def expand_query(query_tokens, max_syns=2):
    """
    Expand query tokens using WordNet synonyms.

    Parameters
    ----------
    query_tokens : list of str
        List of words in the query.
    max_syns : int
        Max synonyms to add per word.

    Returns
    -------
    list of str
        Expanded query tokens.
    """
    expanded = set(query_tokens)

    for word in query_tokens:
        synsets = wordnet.synsets(word)
        added = 0
        for syn in synsets:
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower() and lemma_name.isalpha():
                    expanded.add(lemma_name)
                    added += 1
                if added >= max_syns:
                    break
            if added >= max_syns:
                break

    return list(expanded)

def expand_query_pos_filtered(query_tokens, max_syns=2):
    from nltk.corpus import wordnet
    from nltk import pos_tag

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        else:
            return None

    expanded = set(query_tokens)
    tagged = pos_tag(query_tokens)

    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        if wn_pos is None:
            continue
        synsets = wordnet.synsets(word, pos=wn_pos)
        added = 0
        for syn in synsets:
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower() and lemma_name.isalpha():
                    expanded.add(lemma_name)
                    added += 1
                if added >= max_syns:
                    break
            if added >= max_syns:
                break

    return list(expanded)