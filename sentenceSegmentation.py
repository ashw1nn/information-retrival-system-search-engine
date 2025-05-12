from util import *

# Add your import statements here
import re
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download("punkt")
# nltk.download("punkt_tab")

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here

		########### Checks for capitalization #####################
		# pattern = r"\b.*?[.?!](?=\s[A-Z]|\Z)"
		# segmentedText = re.findall(pattern, text)
		###########################################################

		abbreviations = [
			"Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "Gen.", "Lt.", "Col.", "Maj.", "Capt.", "Adm.", "Sgt.",
			"Ave.", "Blvd.", "St.", "Rd.", "Ln.", "Mt.", "Ft.", "Sq.", "Pl.", "Hwy.", "Ctr.", "Dept.", "Inc.", "Ltd.",
			"Co.", "Corp.", "Univ.", "Vol.", "No.", "pp.", "et al.", "i.e.", "e.g.", "etc.", "vs.", "cf.", "ca.", "al.",
			"a.m.", "p.m.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."
		]
 
		
		sentences = re.split(r'[.?!]', text)
		segmentedText = []
		previous_words = ""
		for sentence in sentences:
			stripped = sentence.strip()
			if stripped:
				if stripped not in abbreviations:
					segmentedText.append(stripped)
					previous_words = ""
				else:
					previous_words += stripped


		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = sent_tokenize(text)

		return segmentedText