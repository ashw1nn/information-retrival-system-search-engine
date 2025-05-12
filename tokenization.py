from util import *

# Add your import statements here
import re

from nltk.tokenize import TreebankWordTokenizer
from sentenceSegmentation import SentenceSegmentation


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for sentence in text:
			words = sentence.split()
			for word in words:
				tokens = re.split(r"(\w+)(['’]\w*)?", word)  # Handles possessives & contractions
				tokens = [t for t in tokens if t]  # Remove empty strings
				tokenizedText.extend(tokens)

		tokenizedText = [tokenizedText]
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		tokenizer = TreebankWordTokenizer()
		for sent in text:
			tokenizedText.append(tokenizer.tokenize(sent))

		return tokenizedText