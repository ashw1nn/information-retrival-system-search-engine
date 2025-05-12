# First Install Dependencies

Run the following code:

> pip install -r requirements.txt

# To run the program

Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)]
               [-model MODEL TYPE (tfidf|bm25|lsa)]
               [--no-expand(optional:Doesn't query expand)]
               [--no-rerank(optional:Doesn't use sbert to rerank)]
               [--no-spell(optional:Doesn't incorporate query spell checks)]

Note:

> Make sure you have an OUTPUT_FOLDER and DATASET_FOLDER already created

When the -custom flag is passed, the system will take a query from the user as input. For example:

> python main.py -custom
> Enter query below
> Papers on Aerodynamics
> This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.
