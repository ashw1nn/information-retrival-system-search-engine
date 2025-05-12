import random
import json
from tqdm import tqdm
import ir_datasets

def convert_to_cranfield(dataset_name, output_prefix):
    dataset = ir_datasets.load(dataset_name)

    # Convert documents
    docs = []
    for doc in tqdm(dataset.docs_iter(), desc="Processing documents"):
        docs.append({
            "id": doc.doc_id,
            "body": getattr(doc, 'text', '')
        })
    with open(f"{output_prefix}_docs.json", "w") as f:
        json.dump(docs, f, indent=2)

    # Convert queries — only if they exist
    queries = []
    if hasattr(dataset, "queries_iter"):
        for i, query in enumerate(tqdm(dataset.queries_iter(), desc="Processing queries"), start=1):
            queries.append({
                "query number": str(i),
                "query": getattr(query, 'text', '')
            })
        with open(f"{output_prefix}_queries.json", "w") as f:
            json.dump(queries, f, indent=2)

    # Convert qrels — only if they exist
    qrels = []
    if hasattr(dataset, "qrels_iter"):
        has_qrels = False
        for qrel in tqdm(dataset.qrels_iter(), desc="Processing qrels"):
            has_qrels = True
            qrels.append({
                "query_num": qrel.query_id,
                "id": qrel.doc_id,
                "position": qrel.relevance
            })
        if has_qrels:
            with open(f"{output_prefix}_qrels.json", "w") as f:
                json.dump(qrels, f, indent=2)
        else:
            print(f"[WARN] No qrels found for dataset: {dataset_name}")

def convert_msmarco_subset(dataset_name, output_prefix, num_queries=10000, num_distractor_docs=5):
    dataset = ir_datasets.load(dataset_name)

    # Build lookup for qrels
    qrels_map = {}
    for qrel in dataset.qrels_iter():
        qrels_map.setdefault(qrel.query_id, []).append(qrel.doc_id)

    # Pick N queries with qrels
    queries = list(dataset.queries_iter())
    random.shuffle(queries)
    selected_queries = []
    selected_qrels = []
    selected_doc_ids = set()

    for query in tqdm(queries, desc="Selecting queries"):
        if query.query_id in qrels_map:
            selected_queries.append({
                "query number": query.query_id,
                "query": query.text
            })
            for doc_id in qrels_map[query.query_id]:
                selected_qrels.append({
                    "query_num": query.query_id,
                    "id": doc_id,
                    "position": 1  # binary relevance
                })
                selected_doc_ids.add(doc_id)

            if len(selected_queries) >= num_queries:
                break

    # Add distractors (non-relevant docs) randomly
    all_docs = list(dataset.docs_iter())
    random.shuffle(all_docs)
    for doc in all_docs:
        if len(selected_doc_ids) > (num_queries * (1 + num_distractor_docs)):
            break
        if doc.doc_id not in selected_doc_ids:
            selected_doc_ids.add(doc.doc_id)

    # Save filtered docs
    doc_dict = {doc.doc_id: doc.text for doc in all_docs if doc.doc_id in selected_doc_ids}
    final_docs = [{"id": doc_id, "title": "", "body": body} for doc_id, body in doc_dict.items()]
    
    # Save to disk
    with open(f"{output_prefix}_docs.json", "w") as f:
        json.dump(final_docs, f, indent=2)
    with open(f"{output_prefix}_queries.json", "w") as f:
        json.dump(selected_queries, f, indent=2)
    with open(f"{output_prefix}_qrels.json", "w") as f:
        json.dump(selected_qrels, f, indent=2)


# convert_to_cranfield("disks45/nocr/trec-robust-2004", "trec_robust04")
# convert_to_cranfield("msmarco-passage", "msmarco_passage")
# convert_to_cranfield("beir/scifact", "scifact")

# try:
#     convert_to_cranfield("msmarco-passage/train", "msmarco_passage")
# except Exception as e:
#     print(f"MS MARCO conversion failed: {e}")

try:
    convert_to_cranfield("beir/nfcorpus/test", "nfcorpus")
except Exception as e:
    print(f"nfcorpus conversion failed: {e}")

# try:
#     convert_msmarco_subset("msmarco-passage/train", "msmarco_small", num_queries=10000)
# except Exception as e:
#     print(f"Desizing MSMACRO failed: {e}")