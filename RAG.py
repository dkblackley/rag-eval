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

"""
RAG generation + Ragas evaluation for one retrieval run.

Built to run over *all* queries of a dataset, which for msmarco/fever means
~7k queries at ~1 min of 70b-judge time each, i.e. far longer than one slurm
job. So every stage is resumable:

  * predictions are saved every --save-every answers and resumed from;
  * Ragas runs in chunks of --eval-chunk queries, each chunk appended to
    --output-csv, and queries already in the CSV are skipped on restart;
  * with --deadline (unix time), no new chunk is started when it would not
    finish in time, and the process exits with EXIT_INCOMPLETE (75) so the
    slurm wrapper can resubmit itself.

When every selected query is in the CSV, the means are written to --metadata
and an empty "<output-csv>.done" marker is created.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# Increase CSV field limit for large documents
csv.field_size_limit(sys.maxsize)

EXIT_INCOMPLETE = 75   # EX_TEMPFAIL: ran out of time, resubmit to continue


def update_json_file(filename, new_data):  # opens and writes to a json
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    for key, value in new_data.items():
        data[str(key)] = str(value)
    tmp = f"{filename}.tmp{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, filename)


def atomic_json_dump(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp, path)


# ==========================================
# Shared data loading
# ==========================================
def load_queries(queries_path):
    """
    Returns: dict {qid: question_text}, insertion-ordered as in the file.
    Accepts TSV (qid\\ttext) or JSONL with either "_id" (BEIR) or "id" (msmarco jsonl).
    """
    print(f"Loading queries from {queries_path}...")
    questions_map = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if queries_path.endswith('.jsonl'):
                if not line.strip():
                    continue
                data = json.loads(line)
                qid = str(data.get('_id', data.get('id', '')))
                questions_map[qid] = data.get('text', '')
            else:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2:
                    questions_map[parts[0]] = parts[1]
    print(f"  {len(questions_map)} queries")
    return questions_map


def load_qrels_qids(qrels_path):
    """
    Query ids present in a qrels file. Handles BEIR (query-id corpus-id score,
    with header) and TREC (qid 0 docid rel) layouts: only column 1 is used.
    """
    qids = set()
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] in ('query-id', 'qid', 'query_id'):
                continue
            qids.add(parts[0])
    print(f"  {len(qids)} judged queries in {qrels_path}")
    return qids


def load_run(filepath):
    """
    Ranked results of one retrieval run.
      * .json : {qid: [docid, ...]} in rank order (results_reRank.json)
      * other : TSV qid \\t docid \\t rank, already sorted by rank
    Returns: dict {qid: [docid1, docid2, ...]}
    """
    print(f"Loading ranked results from {filepath}...")
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            raw = json.load(f)
        return {str(q): [str(d) for d in docs] for q, docs in raw.items()}

    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            results.setdefault(parts[0], []).append(parts[1])
    return results


# ==========================================
# Retrieval
# ==========================================
class Retrieval:
    def __init__(self, run_results, corpus_file, needed_qids, top_k=10,
                 max_doc_chars=0, doc_cache=None):
        """
        Args:
            run_results: {qid: [docid, ...]} from load_run
            corpus_file: corpus .jsonl ({"_id", "title", "text"}) or .tsv (docid \\t text)
            needed_qids: only the documents these queries retrieve are loaded,
                         so a 46 GB corpus (msmarco-v2) costs a scan, not RAM
            top_k: how many documents to hand the generator per query
            max_doc_chars: truncate each passage to this many characters (0 = off),
                           keeps 10 passages inside the generator/judge context
            doc_cache: optional jsonl path; the loaded subset is written there and
                       re-used on restart instead of scanning the corpus again
        """
        self.top_k = top_k
        self.max_doc_chars = max_doc_chars
        self.run_results = run_results

        needed = set()
        for q in needed_qids:
            needed.update(run_results.get(q, [])[:top_k])
        print(f"{len(needed)} distinct documents needed for {len(needed_qids)} queries.")

        self.corpus = self._load_cache(doc_cache, needed) if doc_cache else None
        if self.corpus is None:
            print(f"Scanning corpus {corpus_file} for them...")
            self.corpus = self._load_corpus(corpus_file, needed)
            if doc_cache:
                self._write_cache(doc_cache)
        missing = len(needed) - len(self.corpus)
        if missing:
            print(f"Warning: {missing} retrieved documents are not in the corpus.")
        print("Retrieval System Ready.")

    @staticmethod
    def _load_cache(path, needed):
        if not os.path.exists(path):
            return None
        corpus = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                corpus[d['_id']] = d['text']
        if not needed.issubset(corpus):
            print(f"Doc cache {path} does not cover this query set; rescanning.")
            return None
        print(f"Loaded {len(corpus)} documents from cache {path}")
        return {k: corpus[k] for k in needed}

    def _write_cache(self, path):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, 'w', encoding='utf-8') as f:
            for docid, text in self.corpus.items():
                f.write(json.dumps({'_id': docid, 'text': text}) + '\n')
        os.replace(tmp, path)
        print(f"Wrote {len(self.corpus)} documents to cache {path}")

    @staticmethod
    def _load_corpus(filepath, needed):
        """
        Streams the corpus and keeps only `needed` docids.
        Returns: dict {docid: text}
        """
        corpus = {}
        remaining = len(needed)
        needed_b = {d.encode() for d in needed}
        prefix = b'{"_id": "'
        is_jsonl = filepath.endswith('.jsonl')
        with open(filepath, 'rb') as f:
            for line in tqdm(f, desc="Scanning corpus", mininterval=30):
                if is_jsonl:
                    # Cheap id check first: json-decode only the lines we keep.
                    if line.startswith(prefix):
                        end = line.find(b'"', len(prefix))
                        if line[len(prefix):end] not in needed_b:
                            continue
                    data = json.loads(line)
                    docid = str(data.get('_id', ''))
                    if docid not in needed:
                        continue
                    title = data.get('title', '') or ''
                    text = data.get('text', '') or ''
                    corpus[docid] = f"{title} {text}".strip()
                else:
                    tab = line.find(b'\t')
                    if tab < 0 or line[:tab] not in needed_b:
                        continue
                    corpus[line[:tab].decode()] = line[tab + 1:].decode('utf-8', 'replace').strip()
                remaining -= 1
                if remaining == 0:
                    break
        return corpus

    def retrieve(self, query_id):
        """
        Returns list of top_k document texts for the given query_id.
        """
        if query_id not in self.run_results:
            return []

        retrieved_texts = []
        for doc_id in self.run_results[query_id][:self.top_k]:
            text = self.corpus.get(doc_id, "[DOCUMENT NOT FOUND IN CORPUS]")
            if self.max_doc_chars and len(text) > self.max_doc_chars:
                text = text[:self.max_doc_chars]
            retrieved_texts.append(text)
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
        from langchain_ollama import ChatOllama

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
def build_judge(port, num_ctx):
    """Construct the local judge LLM + embeddings and bind them to the metrics."""
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import faithfulness, answer_relevancy

    judge_model = os.environ.get("JUDGE_MODEL", "llama3:70b")
    print(f"Initializing Local Judge ({judge_model} via Ollama) on port {port}...")
    ollama_model = ChatOllama(
        model=judge_model,
        temperature=0,          # Deterministic grading
        num_ctx=num_ctx,
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

    return judge_llm, judge_embeddings, [faithfulness, answer_relevancy]


# ==========================================
# Stage 1
# ==========================================
def run_generation(retriever, questions_map, qids, output_path, generator_model,
                   num_ctx, port, workers, save_every, deadline, margin):
    """
    Generates answers for `qids` and saves them to output_path.
    Resumes: any qid already present in output_path is kept and not regenerated.
    Returns (answers, complete).
    """
    output_answers = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            output_answers = json.load(f)
        print(f"Found {len(output_answers)} existing predictions in {output_path}")

    todo = [q for q in qids if q not in output_answers]
    if not todo:
        print("All predictions already present; skipping generation.")
        return output_answers, True

    print(f"Generating {len(todo)} of {len(qids)} answers with {workers} worker(s)...")
    generator = RAGGenerator(model_name=generator_model, num_ctx=num_ctx, port=port)

    def _one(qid):
        return qid, generator.generate(questions_map[qid], retriever.retrieve(qid))

    complete = True
    done_since_save = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # Submit lazily so a deadline stop doesn't leave thousands of queued calls.
        it = iter(todo)
        pending = set()
        for _ in range(max(1, workers) * 2):
            q = next(it, None)
            if q is None:
                break
            pending.add(pool.submit(_one, q))

        bar = tqdm(total=len(todo), desc="Generating", mininterval=30)
        while pending:
            fut = next(as_completed(pending))
            pending.remove(fut)
            qid, answer = fut.result()
            output_answers[qid] = answer
            bar.update(1)
            done_since_save += 1
            if done_since_save >= save_every:
                atomic_json_dump(output_answers, output_path)
                done_since_save = 0

            if deadline and time.time() > deadline - margin:
                complete = False
                continue          # drain in-flight calls, start no new ones
            q = next(it, None)
            if q is not None:
                pending.add(pool.submit(_one, q))
        bar.close()

    complete = complete and all(q in output_answers for q in qids)

    atomic_json_dump(output_answers, output_path)
    print(f"Predictions saved to {output_path} ({len(output_answers)} total)")
    return output_answers, complete


# ==========================================
# Stage 2
# ==========================================
def evaluated_qids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["qid"], dtype={"qid": str})
    except (ValueError, pd.errors.EmptyDataError):
        return set()
    return set(df["qid"].dropna())


def build_dataset(qids, predictions, questions_map, retriever):
    from datasets import Dataset

    data_points = {"user_input": [], "response": [], "retrieved_contexts": []}
    kept = []
    for qid in qids:
        docs = retriever.retrieve(qid)
        if not docs or qid not in predictions:
            continue
        data_points["user_input"].append(questions_map[qid])
        data_points["response"].append(predictions[qid])
        data_points["retrieved_contexts"].append(docs)
        kept.append(qid)
    return Dataset.from_dict(data_points), kept


def run_evaluation(qids, predictions, questions_map, retriever, args, deadline, margin):
    """
    Evaluates `qids` in chunks, appending each chunk to args.output_csv.
    Returns True when every qid has a row in the CSV.
    """
    from ragas import evaluate
    from ragas.run_config import RunConfig

    done = evaluated_qids(args.output_csv)
    todo = [q for q in qids if q not in done]
    print(f"\n{len(done)} queries already evaluated in {args.output_csv}; {len(todo)} to go.")
    if not todo:
        return True

    judge_llm, judge_embeddings, metrics = build_judge(args.ollama_port, args.judge_num_ctx)
    run_config = RunConfig(timeout=args.timeout, max_workers=args.max_workers)

    chunk_times = []
    n_chunks = (len(todo) + args.eval_chunk - 1) // args.eval_chunk
    for c in range(n_chunks):
        if deadline:
            est = max(chunk_times) if chunk_times else 0
            if time.time() + est > deadline - margin:
                print(f"Stopping before chunk {c + 1}/{n_chunks}: "
                      f"{(deadline - time.time()) / 60:.0f} min left, last chunk took {est / 60:.0f} min.")
                return False

        chunk = todo[c * args.eval_chunk:(c + 1) * args.eval_chunk]
        ds, kept = build_dataset(chunk, predictions, questions_map, retriever)
        if not kept:
            continue
        t0 = time.time()
        print(f"Evaluating chunk {c + 1}/{n_chunks} ({len(kept)} queries)...", flush=True)
        results = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )
        df = results.to_pandas()
        df.insert(0, "qid", kept)
        write_header = not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0
        df.to_csv(args.output_csv, mode='a', header=write_header, index=False)
        chunk_times.append(time.time() - t0)
        print(f"  chunk {c + 1} done in {chunk_times[-1]:.0f}s "
              f"({chunk_times[-1] / len(kept):.1f}s/query)", flush=True)

    return all(q in evaluated_qids(args.output_csv) for q in qids)


def summarise(args, qids):
    df = pd.read_csv(args.output_csv, dtype={"qid": str})
    df = df.drop_duplicates("qid", keep="last")
    df = df[df["qid"].isin(set(qids))]

    summary = {}
    for col in ("faithfulness", "answer_relevancy"):
        if col not in df.columns:
            print(f"Warning: no column {col!r}; columns = {list(df.columns)}")
            summary[col] = 0.0
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        summary[col] = float(s.mean()) if s.notna().any() else 0.0
        summary[f"{col}_nan"] = int(s.isna().sum())
    summary["rag_num_queries"] = int(len(df))
    summary["rag_generator"] = os.environ.get("GENERATOR_MODEL", args.generator_model or "qwen2.5:7b-instruct")
    summary["rag_judge"] = os.environ.get("JUDGE_MODEL", "llama3:70b")
    return summary


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Run RAG generation and Ragas evaluation in one pass.")
    p.add_argument("--reranked-file", default="step4_reranked_output.tsv",
                   help="Ranked results: results_reRank.json ({qid: [docid,...]}) or TSV (qid\\tdocid\\trank).")
    p.add_argument("--corpus-file", default="collection.tsv",
                   help="Path to corpus/collection TSV or JSONL.")
    p.add_argument("--queries-path", default="queries.dev.small.tsv",
                   help="Path to queries TSV (qid\\ttext) or JSONL.")
    p.add_argument("--qrels", default=None,
                   help="Only evaluate queries present in this qrels file (e.g. scifact test split).")
    p.add_argument("--predictions-file", default="rag_predictions.json",
                   help="Where the interim predictions are written and resumed from.")
    p.add_argument("--output-csv", default="rag_evaluation_results_local.csv",
                   help="Per-query evaluation results; appended chunk by chunk and resumed from.")
    p.add_argument("--metadata", default="metadata.json",
                   help="Path to metadata file.")
    p.add_argument("--doc-cache", default=None,
                   help="jsonl cache of the documents this run needs (skips the corpus scan on restart).")
    p.add_argument("--limit", type=int, default=0,
                   help="Max number of queries to process. 0 or negative = all.")
    p.add_argument("--top-k", type=int, default=10,
                   help="Documents passed to the generator per query.")
    p.add_argument("--max-doc-chars", type=int, default=2000,
                   help="Truncate each passage to this many characters (0 = no truncation).")
    p.add_argument("--generator-model", default=None,
                   help="Ollama model used to generate answers. Defaults to $GENERATOR_MODEL, "
                        "then qwen2.5:7b-instruct.")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="Generator context window in tokens.")
    p.add_argument("--judge-num-ctx", type=int, default=8192,
                   help="Judge context window in tokens.")
    p.add_argument("--timeout", type=int, default=600,
                   help="Timeout in seconds per LLM/embedding call during evaluation.")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Concurrent Ragas workers (match OLLAMA_NUM_PARALLEL).")
    p.add_argument("--gen-workers", type=int, default=None,
                   help="Concurrent generation requests. Defaults to --max-workers.")
    p.add_argument("--eval-chunk", type=int, default=50,
                   help="Queries per Ragas chunk; each chunk is checkpointed to the CSV.")
    p.add_argument("--save-every", type=int, default=100,
                   help="Save predictions every N generated answers.")
    p.add_argument("--deadline", type=float, default=0,
                   help="Unix time by which to stop; exits with code 75 if work remains.")
    p.add_argument("--deadline-margin", type=float, default=900,
                   help="Seconds before --deadline to stop starting new work.")
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Port of the Ollama server for this task.")
    p.add_argument("--regenerate", action="store_true",
                   help="Discard existing predictions AND evaluation CSV, start from scratch.")
    p.add_argument("--restart-eval", action="store_true",
                   help="Discard the existing evaluation CSV (keep predictions).")
    p.add_argument("--skip-eval", action="store_true",
                   help="Stop after generation; don't run Ragas.")
    args = p.parse_args()

    started = time.time()
    done_marker = f"{args.output_csv}.done"

    if args.regenerate or args.restart_eval:
        doomed = [args.output_csv, done_marker]
        if args.regenerate:
            doomed.append(args.predictions_file)
        for path in doomed:
            if os.path.exists(path):
                print(f"Removing {path}")
                os.remove(path)

    # 1. Which queries: in the queries file, in the run, (in qrels), first --limit
    questions_map = load_queries(args.queries_path)
    run_results = load_run(args.reranked_file)
    qids = [q for q in questions_map if q in run_results]
    if args.qrels:
        judged = load_qrels_qids(args.qrels)
        qids = [q for q in qids if q in judged]
    if args.limit and args.limit > 0:
        qids = qids[:args.limit]
    print(f"{len(qids)} queries selected ({len(run_results)} in run, {len(questions_map)} in queries file).")
    if not qids:
        print("ERROR: no queries selected; check that the query ids match the run/qrels.")
        sys.exit(1)

    retriever = Retrieval(
        run_results,
        args.corpus_file,
        needed_qids=qids,
        top_k=args.top_k,
        max_doc_chars=args.max_doc_chars,
        doc_cache=args.doc_cache,
    )
    print(f"Data loaded at {time.time() - started:.0f}s")

    # 2. Generate (resumes from the predictions file if it already has answers)
    predictions, gen_complete = run_generation(
        retriever,
        questions_map,
        qids,
        args.predictions_file,
        args.generator_model,
        args.num_ctx,
        args.ollama_port,
        args.gen_workers or args.max_workers,
        args.save_every,
        args.deadline,
        args.deadline_margin,
    )
    print(f"Generation stage done at {time.time() - started:.0f}s")
    if not gen_complete:
        print("Out of time during generation; resubmit to continue.")
        sys.exit(EXIT_INCOMPLETE)

    if args.skip_eval:
        print("--skip-eval set; stopping after generation.")
        sys.exit(0)

    # 3. Evaluate in checkpointed chunks
    complete = run_evaluation(qids, predictions, questions_map, retriever, args,
                              args.deadline, args.deadline_margin)
    print(f"Evaluation stage stopped at {time.time() - started:.0f}s (complete={complete})")
    if not complete:
        print("Out of time during evaluation; resubmit to continue.")
        sys.exit(EXIT_INCOMPLETE)

    # 4. Aggregate over the full CSV
    final_scores = summarise(args, qids)
    print(f"\nUpdating {args.metadata} with {final_scores}")
    update_json_file(args.metadata, final_scores)
    with open(done_marker, 'w') as f:
        f.write(f"{final_scores['rag_num_queries']}\n")
    print(f"Total time: {time.time() - started:.0f}s")
