# --- vertexai shim: ragas imports a module langchain-community >=0.4.2 removed ---
import sys as _sys, types as _types
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI as _probe  # noqa: F401
except Exception:
    import langchain_community.chat_models as _cm
    import langchain_community.llms as _lm

    class ChatVertexAI: pass          # isinstance target only, never instantiated
    class VertexAI: pass
    class VertexAIModelGarden: pass

    _m = _types.ModuleType("langchain_community.chat_models.vertexai")
    _m.ChatVertexAI = ChatVertexAI
    _sys.modules["langchain_community.chat_models.vertexai"] = _m
    _cm.vertexai = _m; _cm.ChatVertexAI = ChatVertexAI

    _m2 = _types.ModuleType("langchain_community.llms.vertexai")
    _m2.VertexAI = VertexAI; _m2.VertexAIModelGarden = VertexAIModelGarden
    _sys.modules["langchain_community.llms.vertexai"] = _m2
    _lm.vertexai = _m2; _lm.VertexAI = VertexAI; _lm.VertexAIModelGarden = VertexAIModelGarden
# --- end shim ---

import argparse
import csv
import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Increase CSV field limit for large MS MARCO documents
csv.field_size_limit(sys.maxsize)


def update_json_file(filename, new_data):  # opens and writes to a json
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    for key, value in new_data.items():
        data[str(key)] = str(value)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


# ==========================================
# Retrieval
# ==========================================
class Retrieval:
    def __init__(self, reranked_file="step4_reranked_output.tsv", corpus_file="collection.tsv", top_k=10):
        """
        Args:
            reranked_file: Path to the .tsv file saved in step 4 (qid, docid, rank)
            corpus_file: Path to standard MS MARCO collection.tsv (docid, text)
            top_k: How many documents to hand the generator per query
        """
        self.top_k = top_k

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
                    results[qid].append(docid)
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Please run the reranker step first.")
        return results

    def _load_corpus(self, filepath):
        """
        Parses standard MS MARCO collection (.tsv) or SciFact/TREC (.jsonl)
        Returns: dict {docid: text}
        """
        corpus = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Iterate line by line for memory efficiency
                for line in tqdm(f, desc="Loading Corpus"):
                    if filepath.endswith('.jsonl'):
                        data = json.loads(line)
                        docid = str(data.get('_id', ''))
                        # SciFact has a title, MS MARCO doesn't. Combine if present.
                        title = data.get('title', '')
                        text = data.get('text', '')
                        corpus[docid] = f"{title} {text}".strip()
                    else:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            docid = parts[0]
                            text = parts[1].strip()
                            corpus[docid] = text
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Ensure corpus file is present.")
        return corpus

    def retrieve(self, query_id):
        """
        Returns list of top_k document texts for the given query_id.
        """
        if query_id not in self.run_results:
            return []

        top_doc_ids = self.run_results[query_id][:self.top_k]

        # Map IDs to Text
        retrieved_texts = []
        for doc_id in top_doc_ids:
            if doc_id in self.corpus:
                retrieved_texts.append(self.corpus[doc_id])
            else:
                retrieved_texts.append("[DOCUMENT NOT FOUND IN CORPUS]")
                print(f"Warning: Document {doc_id} not found in corpus of length {len(self.corpus)}")

        return retrieved_texts


# ==========================================
# Generation
# ==========================================
class RAGGenerator:
    def __init__(self, model_name=None, num_ctx=8192, port=11434):
        """
        Answer generator, served by the same Ollama instance as the judge.
        model_name falls back to $GENERATOR_MODEL so the slurm/bash scripts
        can swap in a small model for testing without editing this file.
        """
        self.model_name = model_name or os.environ.get("GENERATOR_MODEL", "qwen2.5:7b-instruct")
        print(f"Loading Generator: {self.model_name} via Ollama on port {port} (num_ctx={num_ctx})...")
        self.model = ChatOllama(
            model=self.model_name,
            temperature=0,          # Deterministic generation
            num_ctx=num_ctx,
            num_predict=128,
            base_url=f"http://127.0.0.1:{port}",
        )

    def generate(self, query, retrieved_docs):
        """
        Generates an answer using the query and the top_k retrieved documents.
        """
        if not retrieved_docs:
            return "No documents retrieved."

        # Number the passages so the model can distinguish them.
        context = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(retrieved_docs))
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say so. "
            "Answer in one or two sentences, with no preamble.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        return self.model.invoke(prompt).content.strip()


# ==========================================
# Judge
# ==========================================
def build_judge(port):
    """Construct the local judge LLM + embeddings and bind them to the metrics."""
    judge_model = os.environ.get("JUDGE_MODEL", "llama3:70b")
    print(f"Initializing Local Judge ({judge_model} via Ollama) on port {port}...")
    ollama_model = ChatOllama(
        model=judge_model,
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


# ==========================================
# Shared data loading
# ==========================================
def load_queries(queries_path):
    """
    Loads queries once, for both stages.
    Returns: dict {qid: question_text}, insertion-ordered as in the file.
    """
    print(f"Loading queries from {queries_path}...")
    questions_map = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if queries_path.endswith('.jsonl'):
                data = json.loads(line)
                qid = str(data.get('_id', ''))
                questions_map[qid] = data.get('text', '')
            else:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    questions_map[parts[0]] = parts[1]
    return questions_map


# ==========================================
# Stage 1
# ==========================================
def run_generation(retriever, questions_map, limit, output_path, generator_model, num_ctx, port):
    """
    Generates answers for the first `limit` queries and saves them to output_path.
    Resumes: any qid already present in output_path is kept and not regenerated.
    """
    output_answers = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            output_answers = json.load(f)
        print(f"Found {len(output_answers)} existing predictions in {output_path}")

    qids = list(questions_map.keys())
    if limit and limit > 0:
        qids = qids[:limit]

    todo = [q for q in qids if q not in output_answers]
    if not todo:
        print("All predictions already present; skipping generation.")
        return output_answers

    print(f"Generating {len(todo)} of {len(qids)} answers...")
    generator = RAGGenerator(model_name=generator_model, num_ctx=num_ctx, port=port)

    for i, qid in enumerate(tqdm(todo, desc="Generating")):
        query_text = questions_map[qid]

        # A. Retrieve
        top_docs = retriever.retrieve(qid)

        # B. Generate
        answer = generator.generate(query_text, top_docs)

        # Store
        output_answers[qid] = answer

        # Save interim results so a crash or a Ctrl-C doesn't lose the work
        if (i + 1) % 100 == 0:
            with open(output_path, 'w') as f:
                json.dump(output_answers, f, indent=4)

    with open(output_path, 'w') as f:
        json.dump(output_answers, f, indent=4)
    print(f"Predictions saved to {output_path}")

    return output_answers


# ==========================================
# Stage 2
# ==========================================
def prepare_dataset(predictions, questions_map, retriever):
    """
    Combines Questions + Answers + Retrieved Contexts into a Hugging Face Dataset.
    Uses the retriever that is already in memory from stage 1.
    """
    data_points = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": []
    }
    qids = []

    print("Re-constructing evaluation dataset...")
    for qid, answer_text in predictions.items():
        if qid not in questions_map:
            continue

        question_text = questions_map[qid]
        retrieved_docs = retriever.retrieve(qid)
        if not retrieved_docs:
            continue

        data_points["user_input"].append(question_text)
        data_points["response"].append(answer_text)
        data_points["retrieved_contexts"].append(retrieved_docs)
        qids.append(qid)

    return Dataset.from_dict(data_points), qids


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Run RAG generation and Ragas evaluation in one pass.")
    p.add_argument("--reranked-file", default="step4_reranked_output.tsv",
                   help="Path to reranked output TSV used by Retrieval.")
    p.add_argument("--corpus-file", default="collection.tsv",
                   help="Path to corpus/collection TSV or JSONL.")
    p.add_argument("--queries-path", default="queries.dev.small.tsv",
                   help="Path to queries TSV (qid\\ttext) or JSONL.")
    p.add_argument("--predictions-file", default="rag_predictions.json",
                   help="Where the interim predictions are written and resumed from.")
    p.add_argument("--output-csv", default="rag_evaluation_results_local.csv",
                   help="Where to write per-query evaluation results as CSV.")
    p.add_argument("--metadata", default="metadata.json",
                   help="Path to metadata file.")
    p.add_argument("--limit", type=int, default=100,
                   help="Max number of queries to process. Use 0 or a negative value to run all.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Documents passed to the generator per query.")
    p.add_argument("--generator-model", default=None,
                   help="Ollama model used to generate answers. Defaults to $GENERATOR_MODEL, "
                        "then qwen2.5:7b-instruct.")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="Generator context window in tokens. scifact top-5 has p99 ~3.5k.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout in seconds per LLM/embedding call during evaluation.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Number of concurrent evaluation workers. Use 1 for sequential.")
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Port of the Ollama server for this task (see PORT in run_eval.slurm).")
    p.add_argument("--regenerate", action="store_true",
                   help="Ignore any existing predictions file and generate from scratch.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Stop after generation; don't run Ragas.")
    args = p.parse_args()

    started = time.time()

    if args.regenerate and os.path.exists(args.predictions_file):
        os.remove(args.predictions_file)

    # 1. Load queries and corpus ONCE, shared by both stages
    questions_map = load_queries(args.queries_path)

    retriever = Retrieval(
        reranked_file=args.reranked_file,
        corpus_file=args.corpus_file,
        top_k=args.top_k,
    )

    # 2. Generate (resumes from the predictions file if it already has answers)
    predictions = run_generation(
        retriever,
        questions_map,
        args.limit,
        args.predictions_file,
        args.generator_model,
        args.num_ctx,
        args.ollama_port,
    )
    print(f"Generation stage done at {time.time() - started:.0f}s")

    if args.skip_eval:
        print("--skip-eval set; stopping after generation.")
        sys.exit(0)

    # 3. Build the judge and the evaluation dataset
    judge_llm, judge_embeddings = build_judge(args.ollama_port)

    ragas_dataset, qids = prepare_dataset(predictions, questions_map, retriever)
    print(f"\nDataset ready with {len(ragas_dataset)} samples.")

    my_run_config = RunConfig(
        timeout=args.timeout,            # Wait up to 600 seconds (10 mins) per call
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

    # 4. Save Results
    df = results.to_pandas()
    df.insert(0, "qid", qids)

    df.to_csv(args.output_csv, index=False)          # write before aggregating
    print(f"\nDetailed per-query results saved to {args.output_csv}")
    print(f"Result columns: {list(df.columns)}")

    def _mean(metric, fallback):
        col = getattr(metric, "name", None) or fallback
        if col not in df.columns:
            print(f"Warning: no column {col!r}; columns = {list(df.columns)}")
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) else 0.0

    final_scores = {
        "faithfulness": _mean(faithfulness, "faithfulness"),
        "answer_relevancy": _mean(answer_relevancy, "answer_relevancy"),
    }

    print(f"\nUpdating {args.metadata} with {final_scores}")
    update_json_file(args.metadata, final_scores)
    print(f"Total time: {time.time() - started:.0f}s")