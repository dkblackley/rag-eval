import argparse

import torch
import csv
import sys
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Increase CSV field limit for large MS MARCO documents
csv.field_size_limit(sys.maxsize)

class Retrieval:
    def __init__(self, reranked_file="step4_reranked_output.tsv", corpus_file="collection.tsv"):
        """
        Args:
            reranked_file: Path to the .tsv file saved in step 4 (qid, docid, rank)
            corpus_file: Path to standard MS MARCO collection.tsv (docid, text)
        """
        print(f"Loading Reranked Results from {reranked_file}...")
        self.run_results = self._load_reranked_results(reranked_file)

        print(f"Loading Corpus from {corpus_file} (This may take memory)...")
        self.corpus = self._load_corpus(corpus_file)
        print("Retrieval System Ready.")

    def _load_reranked_results(self, filepath):
        """
        Parses format: qid \t docid \t rank
        Returns: dict {qid: [docid1, docid2, ...]} (sorted by rank)
        """
        results = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3: continue

                    qid, docid, rank = parts[0], parts[1], parts[2]

                    if qid not in results:
                        results[qid] = []

                    # We assume the file is already sorted by rank as per your save code
                    # If not, we would need to store (docid, rank) and sort later
                    results[qid].append(docid)
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Please run the reranker step first.")
        return results

    def _load_corpus(self, filepath):
        """
        Parses standard MS MARCO collection: docid \t text
        Returns: dict {docid: text}
        """
        corpus = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Iterate line by line for memory efficiency
                for line in tqdm(f, desc="Loading Corpus"):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        docid = parts[0]
                        text = parts[1].strip()
                        corpus[docid] = text
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Ensure MS MARCO collection.tsv is present.")
        return corpus

    def retrieve(self, query_id):
        """
        Returns list of top 5 document texts for the given query_id.
        """
        if query_id not in self.run_results:
            return []

        # Get top 5 Doc IDs
        top_doc_ids = self.run_results[query_id][:5]

        # Map IDs to Text
        retrieved_texts = []
        for doc_id in top_doc_ids:
            if doc_id in self.corpus:
                retrieved_texts.append(self.corpus[doc_id])
            else:
                retrieved_texts.append("[DOCUMENT NOT FOUND IN CORPUS]")
                print(f"Warning: Document {doc_id} not found in corpus of length {len(self.corpus)}")

        return retrieved_texts


class RAGGenerator:
    def __init__(self, model_name="din0s/t5-base-msmarco-nlgen-ob"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Generator: {model_name} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate(self, query, retrieved_docs):
        """
        Generates an answer using the query and the top 5 retrieved documents.
        """
        if not retrieved_docs:
            return "No documents retrieved."

        # Concatenate passages. T5 MS MARCO usually trained on concatenation.
        # Format: "query: <q> context: <p1> <p2> ..."
        context = " ".join(retrieved_docs)
        input_text = f"query: {query} context: {context}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Run RAG over MS MARCO queries and save predictions.")
    p.add_argument(
        "--reranked-file",
        default="step4_reranked_output.tsv",
        help="Path to reranked output TSV used by Retrieval.",
    )
    p.add_argument(
        "--corpus-file",
        default="collection.tsv",
        help="Path to corpus/collection TSV.",
    )
    p.add_argument(
        "--queries-path",
        default="queries.dev.small.tsv",
        help="Path to queries TSV (qid\\ttext).",
    )
    p.add_argument(
        "--output",
        default="rag_predictions.json",
        help="Where to write the JSON predictions.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of queries to process. Use 0 or a negative value to run all.",
    )
    args = p.parse_args()

    # 1. Initialize Retrieval with your specific file paths
    # Ensure 'collection.tsv' exists (Standard MS MARCO collection file)
    retriever = Retrieval(
        reranked_file=args.reranked_file,
        corpus_file=args.corpus_file,
    )

    # 2. Initialize Generator
    generator = RAGGenerator()

    # 3. Load Queries (Standard MS MARCO dev queries format: qid \t text)
    queries_path = args.queries_path
    print(f"Processing queries from {queries_path}...")

    output_answers = {}

    try:
        with open(queries_path, 'r') as f:
            # Limit to first 5 for testing; remove [islice] loop to run all
            from itertools import islice

            for line in islice(f, args.limit):
                qid, query_text = line.strip().split('\t')

                # A. Retrieve
                top_docs = retriever.retrieve(qid)

                # B. Generate
                answer = generator.generate(query_text, top_docs)

                # Store
                output_answers[qid] = answer

                print(f"\nQID: {qid}")
                print(f"Query: {query_text}")
                print(f"Generated Answer: {answer}")

    except FileNotFoundError:
        print(f"Error: {queries_path} not found.")

    # 4. Save Predictions
    import json
    with open(args.output, 'w') as f:
        json.dump(output_answers, f, indent=4)
    print("\nPredictions saved to rag_predictions.json")