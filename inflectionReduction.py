from util import *

# Add your import statements here

from nltk.stem import PorterStemmer


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		#Fill in code here
		stemmer = PorterStemmer()
		for tokens in text:
			stemmed_words = [stemmer.stem(word) for word in tokens]
			reducedText.append(stemmed_words)

		return reducedText


