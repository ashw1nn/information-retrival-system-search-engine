from util import *

# Add your import statements here
import math



class Evaluation():

	def get_true_doc_ids(self, query_id, qrels):
		return [int(rel["id"]) for rel in qrels if int(rel["query_num"]) == int(query_id)]

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in true_doc_IDs]
		precision = len(relevant_retrieved) / k if k else 0.0

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precisions = [
			self.queryPrecision(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]
		meanPrecision = sum(precisions) / len(precisions)

		return meanPrecision
	
	def perQueryPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		return [
			self.queryPrecision(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		if not true_doc_IDs:
			return 0.0
		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = [doc_id for doc_id in retrieved_k if doc_id in true_doc_IDs]
		recall = len(relevant_retrieved) / len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recalls = [
			self.queryRecall(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]
		meanRecall = sum(recalls) / len(recalls)

		return meanRecall
	
	def perQueryRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		return [
			self.queryRecall(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		beta = 0.5
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision + recall == 0:
			return 0.0

		beta_squared = beta ** 2
		fscore = ((1 + beta_squared) * (precision * recall)) / (beta_squared * precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fs = [
			self.queryFscore(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]
		meanFscore = sum(fs) / len(fs)

		return meanFscore

	def perQueryFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		return [
			self.queryFscore(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		 # Build graded relevance map for this query
		rel_dict = {}
		for rel in qrels:
			if int(rel["query_num"]) == int(query_id):
				doc_id = int(rel["id"])
				position = int(rel["position"])
				# Use inverse of position for relevance score (e.g., 1 → 4, 2 → 3, etc.)
				rel_dict[doc_id] = max(0, 5 - position)  # Ensure non-negative

		# DCG calculation
		dcg = 0.0
		for i, doc_id in enumerate(query_doc_IDs_ordered[:k]):
			rel = rel_dict.get(doc_id, 0)
			dcg += rel / math.log2(i + 2)

		# IDCG calculation (ideal ranking)
		ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]
		idcg = sum([rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels)])

		nDCG = dcg / idcg if idcg != 0 else 0.0

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		ndcgs = [
			self.queryNDCG(doc_IDs_ordered[i], query_ids[i], [], qrels, k)
			for i in range(len(query_ids))
		]
		meanNDCG = sum(ndcgs) / len(ndcgs)

		return meanNDCG
	
	def perQueryNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		return [
			self.queryNDCG(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), qrels, k)
			for i in range(len(query_ids))
		]


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		ap = 0.0
		hits = 0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				hits += 1
				ap += hits / (i + 1)
		avgPrecision = ap / len(true_doc_IDs) if true_doc_IDs else 0.0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		maps = [
			self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], q_rels), k)
			for i in range(len(query_ids))
		]
		meanAveragePrecision = sum(maps) / len(maps)

		return meanAveragePrecision
	
	def perQueryAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		return [
			self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], self.get_true_doc_ids(query_ids[i], qrels), k)
			for i in range(len(query_ids))
		]