import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO and WARNING messages


from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

import time
from tqdm import tqdm

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval(args)
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, spell-correct, stem/lemmatize, and remove stopwords
		"""

		# 1. Segment queries
		segmentedQueries = []
		for query in tqdm(queries, desc="Segmenting Queries"):
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))

		# 2. Tokenize + Spell Correct
		tokenizedQueries = []
		for query in tqdm(segmentedQueries, desc="Tokenizing + SpellCorrecting Queries"):
			tokenizedQuery = self.tokenize(query)
			# Flatten for spell correction
			flat = [token for sentence in tokenizedQuery for token in sentence]
			if self.args.spell:
				flat = self.informationRetriever.spell_corrector.correct_query(flat)
			# Rewrap as one "sentence" per query
			tokenizedQueries.append([flat])
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))

		# 3. Inflection Reduction (Stemming)
		reducedQueries = []
		for query in tqdm(tokenizedQueries, desc="Stemming Queries"):
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))

		# 4. Stopword Removal
		stopwordRemovedQueries = []
		for query in tqdm(reducedQueries, desc="Removing Stopwords from Queries"):
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		# Final output
		return stopwordRemovedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in tqdm(docs, desc="Segmenting Documents"):
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in tqdm(segmentedDocs, desc="Tokenizing Documents"):
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tqdm(tokenizedDocs, desc="Stemming Documents"):
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in tqdm(reducedDocs, desc="Removing Stopwords from Documents"):
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""
		
		if args.dataset == "cranfield":
			datasetToBeUsed = "cran"
		else:
			datasetToBeUsed = args.dataset
			

		# Read queries
		queries_json = json.load(open(args.dataset + "/" + datasetToBeUsed + "_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "/" + datasetToBeUsed +  "_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Start measuring time
		start_time = time.time()

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids, raw_docs=docs)
		# Rank the documents for each query
		doc_IDs_ordered = list(tqdm(self.informationRetriever.rank(processedQueries), desc="Ranking Queries"))

		# End measuring time
		end_time = time.time()
		runtime = end_time - start_time
		print(f"IR System Run Time: {runtime:.4f} seconds")

		# Read relevance judements
		qrels = json.load(open(args.dataset + "/" +  datasetToBeUsed + "_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		outRanks = 20
		for k in range(1, outRanks+1):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			# print("Precision, Recall and F-score @ " +  
			# 	str(k) + " : " + str(precision) + ", " + str(recall) + 
			# 	", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			# print("MAP, nDCG @ " +  
			# 	str(k) + " : " + str(MAP) + ", " + str(nDCG))
			print(f"@{k} : Precision:{precision:.4f}, Recall:{recall:.4f}, F-score:{fscore:.4f}, MAP:{MAP:.4f}, nDCG:{nDCG:.4f}")

		# Plot the metrics and save plot 
		plt.plot(range(1, outRanks+1), precisions, label="Precision")
		plt.plot(range(1, outRanks+1), recalls, label="Recall")
		plt.plot(range(1, outRanks+1), fscores, label="F-Score")
		plt.plot(range(1, outRanks+1), MAPs, label="MAP")
		plt.plot(range(1, outRanks+1), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + datasetToBeUsed + "_eval_plot.png")

		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "/cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids, raw_docs=docs)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	# Added by me
	parser.add_argument('--no-expand', dest='expand', action='store_false',
						help="Disable query expansion (enabled by default)")
	parser.add_argument('--no-rerank', dest='rerank', action='store_false',
						help="Disable SBERT reranking (enabled by default)")
	parser.add_argument('--no-spell', dest='spell', action='store_false',
						help="Disable spell correction (enabled by default)")
	parser.add_argument('--use-tfidf', action='store_true',
						help="Use TF-IDF instead of BM25 for initial document scoring")


	# Default values (enabled unless user disables)
	parser.set_defaults(expand=True, rerank=True, spell=True)
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
