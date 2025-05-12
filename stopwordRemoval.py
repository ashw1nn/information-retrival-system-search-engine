from util import *

# Add your import statements here
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		#Fill in code here
		stop_words = set(stopwords.words('english'))
		# print("NLTK stop words length: ", len(stop_words))

		# vectorizer = TfidfVectorizer(stop_words=None, tokenizer=lambda x: x, preprocessor=lambda x: x)
		# tfidf_matrix = vectorizer.fit_transform(text)
		# low_tfidf_words = set(np.array(vectorizer.get_feature_names_out())[np.argsort(np.sum(tfidf_matrix.toarray(), axis=0))[:50]])
		# print("TFIDF stop words length: ", len(low_tfidf_words))


		for tokens in text:
			filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
			stopwordRemovedText.append(filtered_tokens)
			
		return stopwordRemovedText




	