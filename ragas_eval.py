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


def update_json_file(filename, new_data):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    for key, value in new_data.items():
        data[str(key)] = str(value)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# ---------------------------------------------------------
# CONFIGURATION: OPEN SOURCE / LOCAL SETUP
# ---------------------------------------------------------

def build_judge(port):
    """Construct the local judge LLM + embeddings and bind them to the metrics."""
    print(f"Initializing Local Judge (Llama 3 via Ollama) on port {port}...")
    ollama_model = ChatOllama(
        model="llama3:70b",
        temperature=0,          # Deterministic grading
        num_ctx=8192,
        base_url=f"http://127.0.0.1:{port}",
    )
    judge_llm = LangchainLLMWrapper(ollama_model)

    print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    judge_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    faithfulness.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_embeddings

    return judge_llm, judge_embeddings

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
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if queries_file.endswith('.jsonl'):
                data = json.loads(line)
                qid = str(data.get('_id', ''))
                questions_map[qid] = data.get('text', '')
            else:
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
    qids = []

    print("Re-constructing evaluation dataset...")
    for qid, answer_text in predictions.items():
        if qid not in questions_map:
            continue

        question_text = questions_map[qid]
        retrieved_docs = retriever.retrieve(qid)

        data_points["question"].append(question_text)
        data_points["answer"].append(answer_text)
        data_points["contexts"].append(retrieved_docs)
        qids.append(qid)

    return Dataset.from_dict(data_points), qids

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
    p.add_argument(
        "--ollama-port",
        type=int,
        default=11434,
        help="Port of the Ollama server for this task (see PORT in run_eval.slurm).",
    )
    args =  p.parse_args()


    if not os.path.exists(args.predictions_file):
        print(f"Error: {args.predictions_file} missing. Run rag_msmarco.py first.")
        exit()

    judge_llm, judge_embeddings = build_judge(args.ollama_port)

    # 1. Prepare Data
    ragas_dataset, qids = prepare_dataset(args.predictions_file, args.queries_file, args.retrieved_file,
                                          args.collection)
    print(f"\nDataset ready with {len(ragas_dataset)} samples.")

    # 2. Run Evaluation
    # We pass the specific metrics we configured above
    my_run_config = RunConfig(
        timeout=args.timeout,      # Wait up to 600 seconds (10 mins) per call
        max_workers=args.max_workers     # Run ONLY 1 evaluation at a time (Sequential)
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
    df.insert(0, "qid", qids)

    print("\nEvaluation Results:")
    print(results)

    # Filter out NaNs (failed rows) so they don't break the average
    f_scores = [x for x in results["Faithfulness"] if not pd.isna(x)]
    r_scores = [x for x in results["AnswerRelevancy"] if not pd.isna(x)]

    final_scores = {
        # safely calculate average, default to 0 if list is empty
        "faithfulness": sum(f_scores) / len(results["Faithfulness"]) if f_scores else 0,
        "answer_relevancy": sum(r_scores) / len(results["AnswerRelevancy"]) if r_scores else 0
    }

    output_csv = args.output_csv
    df.to_csv(output_csv, index=False)
    print(f"\nDetailed per-query results saved to {output_csv}")

    # Metric column names, taken from the metric objects so they can't drift
    faith_col = faithfulness.name
    rel_col = answer_relevancy.name

    df[faith_col].fillna(0).mean()
    df[rel_col].fillna(0).mean()
    f_scores = df[faith_col].dropna()
    r_scores = df[rel_col].dropna()

    final_scores = {
        "faithfulness": f_scores.mean() if len(f_scores) else 0,
        "answer_relevancy": r_scores.mean() if len(r_scores) else 0,
    }

    print(f"\nUpdating {args.metadata} with {final_scores}")
    update_json_file(args.metadata, final_scores)


