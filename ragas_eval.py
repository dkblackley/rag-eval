import argparse
import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig 
# Import the Retrieval class to fetch contexts
try:
    from rag_msmarco import Retrieval
except ImportError:
    print("Error: rag_msmarco.py not found. Please ensure it is in the same directory.")
    exit()


def update_json_file(filename, new_data): # opens and writes to a json
    with open(filename, 'r+') as f:
        data = json.load(f)
        for key, value in new_data.items():
            data[str(key)] = str(value)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

# ---------------------------------------------------------
# CONFIGURATION: OPEN SOURCE / LOCAL SETUP
# ---------------------------------------------------------

# 1. Initialize the Judge LLM (Ollama)
# We use Llama 3 because it follows grading instructions much better than older models.
# Ensure you have run `ollama pull llama3` in your terminal first.
print("Initializing Local Judge (Llama 3 via Ollama)...")
ollama_model = ChatOllama(
    model="llama3:70b",
    temperature=0, # Deterministic grading
    base_url="http://localhost:11434" # Default Ollama URL
)
# Wrap it so Ragas can use it
judge_llm = LangchainLLMWrapper(ollama_model)

# 2. Initialize Local Embeddings
# Ragas needs embeddings to calculate 'Answer Relevancy' (cosine similarity).
# We use a standard, small, fast HuggingFace model.
print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
)
judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# 3. Assign Models to Metrics
# Faithfulness only needs the LLM to check for hallucinations
faithfulness.llm = judge_llm

# Answer Relevancy needs the LLM (to generate questions) AND Embeddings (to measure similarity)
answer_relevancy.llm = judge_llm
answer_relevancy.embeddings = judge_embeddings

# ---------------------------------------------------------
# DATA PREPARATION & EXECUTION
# ---------------------------------------------------------
# File Paths
PREDICTIONS_FILE = "rag_predictions.json"
QUERIES_FILE = "queries.dev.small.tsv"
RETRIEVED_FILE = "step4_reranked_output.tsv"
COLLECTION_FILE = "collection.tsv"

def prepare_dataset(pred_file, queries_file, retrieved_file, collection_file):
    """
    Combines Questions + Answers + Retrieved Contexts into a Hugging Face Dataset
    """
    print(f"Loading predictions from {pred_file}...")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    print(f"Loading queries from {queries_file}...")
    questions_map = {}
    with open(queries_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                questions_map[parts[0]] = parts[1]

    print("Initializing Retriever to fetch contexts...")
    retriever = Retrieval(reranked_file=retrieved_file, corpus_file=collection_file)

    data_points = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    print("Re-constructing evaluation dataset...")
    for qid, answer_text in predictions.items():
        if qid not in questions_map:
            continue

        question_text = questions_map[qid]
        retrieved_docs = retriever.retrieve(qid)

        data_points["question"].append(question_text)
        data_points["answer"].append(answer_text)
        data_points["contexts"].append(retrieved_docs)

    return Dataset.from_dict(data_points)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Run Ragas evaluation over existing RAG predictions.")
    p.add_argument(
        "--predictions-file",
        default=PREDICTIONS_FILE,
        help="Path to JSON predictions produced by rag_msmarco.py.",
    )
    p.add_argument(
        "--queries-file",
        default=QUERIES_FILE,
        help="Path to queries TSV used to build the evaluation dataset.",
    )
    p.add_argument(
        "--retrieved-file",
        default=RETRIEVED_FILE,
        help="Path to the qids/docids mapping",
    )
    p.add_argument(
        "--collection",
        default=COLLECTION_FILE,
        help="Path to the qids/docids mapping",
    )
    p.add_argument(
        "--output-csv",
        default="rag_evaluation_results_local.csv",
        help="Where to write per-query evaluation results as CSV.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds per LLM/embedding call during evaluation.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent evaluation workers. Use 1 for sequential.",
    )
    p.add_argument(
        "--metadata",
        default="metadata.json",
        help="Path to metadata file.",
    )
    args =  p.parse_args()


    if not os.path.exists(args.predictions_file):
        print(f"Error: {args.predictions_file} missing. Run rag_msmarco.py first.")
        exit()

    # 1. Prepare Data
    ragas_dataset = prepare_dataset(args.predictions_file, args.queries_file, args.retrieved_file, args.collection)
    print(f"\nDataset ready with {len(ragas_dataset)} samples.")

    # 2. Run Evaluation
    # We pass the specific metrics we configured above
    my_run_config = RunConfig(
        timeout=args.timeout,      # Wait up to 600 seconds (10 mins) per call
        max_workers=1     # Run ONLY 1 evaluation at a time (Sequential)
    )

    print("Starting Ragas Evaluation (Local Mode)...")
    results = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=my_run_config
    )

    # 3. Save Results
    df = results.to_pandas()

    print("\nEvaluation Results:")
    print(results)

    # Filter out NaNs (failed rows) so they don't break the average
    f_scores = [x for x in results["faithfulness"] if not pd.isna(x)]
    r_scores = [x for x in results["answer_relevancy"] if not pd.isna(x)]

    final_scores = {
        # safely calculate average, default to 0 if list is empty
        "faithfulness": sum(f_scores) / len(results["faithfulness"]) if f_scores else 0,
        "answer_relevancy": sum(r_scores) / len(results["answer_relevancy"]) if r_scores else 0
    }

    print(f"\nUpdating {args.metadata} with {final_scores}")
    update_json_file(args.metadata, final_scores)


    output_csv = args.output_csv
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed per-query results saved to {output_csv}")

