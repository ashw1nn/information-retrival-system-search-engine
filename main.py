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
from models import run_lsa_model

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
		if args.dataset == "cranfield/":
			datasetToBeUsed = "cran"
		else:
			datasetToBeUsed = args.dataset[:-1]

		queries_json = json.load(open(args.dataset + datasetToBeUsed + "_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], [item["query"] for item in queries_json]
		processedQueries = self.preprocessQueries(queries)

		docs_json = json.load(open(args.dataset + datasetToBeUsed +  "_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
		processedDocs = self.preprocessDocs(docs)

		start_time = time.time()

		self.informationRetriever.buildIndex(processedDocs, doc_ids, raw_docs=docs)
		if self.args.model == "lsa":
			doc_IDs_ordered = list(tqdm(run_lsa_model(processedDocs, processedQueries, doc_ids), desc="Ranking Queries"))
		else:
			self.informationRetriever.buildIndex(processedDocs, doc_ids)
			doc_IDs_ordered = list(tqdm(self.informationRetriever.rank(processedQueries), desc="Ranking Queries"))

		end_time = time.time()
		runtime = end_time - start_time
		print(f"IR System Run Time: {runtime:.4f} seconds")

		qrels = json.load(open(args.dataset +  datasetToBeUsed + "_qrels.json", 'r'))[:]

		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		outRanks = 20
		for k in range(1, outRanks+1):
			precisions.append(self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k))
			recalls.append(self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k))
			fscores.append(self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k))
			MAPs.append(self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k))
			nDCGs.append(self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k))
			print(f"@{k} : Precision:{precisions[-1]:.4f}, Recall:{recalls[-1]:.4f}, F-score:{fscores[-1]:.4f}, MAP:{MAPs[-1]:.4f}, nDCG:{nDCGs[-1]:.4f}")

		plt.plot(range(1, outRanks+1), precisions, label="Precision")
		plt.plot(range(1, outRanks+1), recalls, label="Recall")
		plt.plot(range(1, outRanks+1), fscores, label="F-Score")
		plt.plot(range(1, outRanks+1), MAPs, label="MAP")
		plt.plot(range(1, outRanks+1), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + datasetToBeUsed + "_eval_plot.png")

		# Save per-query scores @k=10
		k = 10
		per_query_metrics = {
			"map": self.evaluator.perQueryAveragePrecision(doc_IDs_ordered, query_ids, qrels, k),
			"ndcg": self.evaluator.perQueryNDCG(doc_IDs_ordered, query_ids, qrels, k),
			"precision": self.evaluator.perQueryPrecision(doc_IDs_ordered, query_ids, qrels, k)
		}
		model_id = self.args.model
		flags = []
		if self.args.rerank:
			flags.append("rerank")
		if self.args.expand:
			flags.append("expand")
		if self.args.spell:
			flags.append("spell")
		flag_str = "_".join(flags) if flags else "plain"
		out_dir = os.path.join(self.args.out_folder, "results")
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"{model_id}_{flag_str}_scores_k10.json")
		with open(out_path, "w") as f:
			json.dump(per_query_metrics, f, indent=2)
		print(f"✅ Saved per-query scores to {out_path}")

		# Save mean MAP@10 for current config
		metrics_path = "output/run_times.csv"
		run_summary = {
			"model": self.args.model,
			"rerank": self.args.rerank,
			"expand": self.args.expand,
			"spell": self.args.spell,
			"runtime_sec": round(runtime, 2),
			"mean_MAP@10": round(MAPs[9], 4)  # k=10 is index 9
		}
		file_exists = os.path.exists(metrics_path)
		with open(metrics_path, "a", newline="") as f:
			import csv
			writer = csv.DictWriter(f, fieldnames=run_summary.keys())
			if not file_exists:
				writer.writeheader()
			writer.writerow(run_summary)

		
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
		if self.args.model == "lsa":
			doc_IDs_ordered = run_lsa_model(processedDocs, processedQuery, doc_ids)
		else:
			self.informationRetriever.buildIndex(processedDocs, doc_ids)
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
	parser.add_argument('-model', default='bm25', choices=['bm25', 'tfidf', 'lsa'],
						help="IR model to use: bm25 or tfidf or lsa")
	parser.add_argument('--no-expand', dest='expand', action='store_false',
						help="Disable query expansion (enabled by default)")
	parser.add_argument('--no-rerank', dest='rerank', action='store_false',
						help="Disable SBERT reranking (enabled by default)")
	parser.add_argument('--no-spell', dest='spell', action='store_false',
						help="Disable spell correction (enabled by default)")

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
