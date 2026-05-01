import os, json, re, torch, gc
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
KBL_DIR   = os.path.join(BASE_DIR, "data", "kbl-rag")
TASKS_DIR = os.path.join(KBL_DIR, "tasks")
MODEL_NAME = "nlpai-lab/KURE-v1"
TOP_K_LIST = [1, 5, 10]

def _load_ngram_list(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_stopwords_txt(path):
    if not os.path.exists(path): return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def apply_clean_text(text, ngrams, stopwords):
    if not text: return ""
    for phrase in ngrams:
        text = text.replace(phrase, " ")
    if stopwords:
        words = text.split()
        text = " ".join([w for w in words if w not in stopwords])
    return re.sub(r"\s+", " ", text).strip()

# =============================================================================
# build_embedding_configs — 검수 후 버전
# 변경점: stopword_판례/candidates_E_intersection.csv → 검수 후 목록/stopwords_단일어_판례_E_intersection_final.txt
#         stopwords_ngram_판례.txt                    → 검수 후 목록/stopwords_ngram_판례_final.txt
# =============================================================================
def build_embedding_configs(base_dir):
    reviewed   = os.path.join(base_dir, "검수 후 목록")
    case_e     = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_E_intersection_final.txt"))
    ngram_case = _load_ngram_list(os.path.join(reviewed, "stopwords_ngram_판례_final.txt"))

    return {
        "00_Baseline":              {"ngrams": [],         "words": set()},
        "01_판례ngram_검수후":       {"ngrams": ngram_case, "words": set()},
        "02_판례E_검수후":           {"ngrams": [],         "words": case_e},
        "03_판례E+판례ngram_검수후": {"ngrams": ngram_case, "words": case_e},
    }

def load_case_relevance_task(task_file):
    path = os.path.join(TASKS_DIR, task_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    queries = []
    corpus_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item        = json.loads(line)
            q_text      = item["query"]
            r_text      = item["retrieved_text"]
            is_relevant = item["is_relevant"]
            if r_text not in corpus_dict:
                corpus_dict[r_text] = f"doc_{len(corpus_dict)}"
            target_doc_id = corpus_dict[r_text]
            gold_ids = [target_doc_id] if is_relevant else []
            queries.append({"query": q_text, "gold_ids": gold_ids})
    corpus = [{"id": vid, "text": txt} for txt, vid in corpus_dict.items()]
    return corpus, queries

def recall_at_k(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    return 1.0 if any(rid in gold_set for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set: return 1.0 / rank
    return 0.0

def run_one(task_file, task_name, cfg_name, config, model, device):
    corpus, queries = load_case_relevance_task(task_file)
    ngrams = config.get("ngrams", [])
    words  = config.get("words", set())

    corpus_texts  = [apply_clean_text(doc["text"], ngrams, words) for doc in corpus]
    corpus_ids    = [doc["id"] for doc in corpus]
    valid_queries = [q for q in queries if q["gold_ids"]]

    all_embs = []
    chunk_size = 5000
    for i in range(0, len(corpus_texts), chunk_size):
        chunk = corpus_texts[i : i + chunk_size]
        emb = model.encode(chunk, batch_size=32, show_progress_bar=False,
                           convert_to_tensor=True).cpu()
        all_embs.append(emb)
        if device == "cuda":  torch.cuda.empty_cache()
        elif device == "mps": torch.mps.empty_cache()
        gc.collect()
    corpus_embeddings = torch.cat(all_embs, dim=0).to(device)

    query_texts = [apply_clean_text(q["query"], ngrams, words) for q in valid_queries]
    query_embeddings = model.encode(query_texts, batch_size=32,
                                    show_progress_bar=False, convert_to_tensor=True)

    all_hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=max(TOP_K_LIST))
    recalls = {k: [] for k in TOP_K_LIST}
    rrs = []
    for q, hits in zip(valid_queries, all_hits):
        retrieved_ids = [corpus_ids[hit["corpus_id"]] for hit in hits]
        for k in TOP_K_LIST:
            recalls[k].append(recall_at_k(retrieved_ids, q["gold_ids"], k))
        rrs.append(reciprocal_rank(retrieved_ids, q["gold_ids"]))

    del corpus_embeddings, query_embeddings
    if device == "cuda":  torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()
    gc.collect()

    return {
        "task":   task_name,
        "설정":   cfg_name,
        **{f"R@{k}": round(np.mean(recalls[k]), 4) for k in TOP_K_LIST},
        "MRR@10": round(np.mean(rrs), 4),
    }

def main():
    print("=" * 70)
    print(f"  Embedding Retrieval Benchmark (판례) — 검수 후 버전")
    print(f"  모델: {MODEL_NAME}")
    print(f"  불용어 경로: 검수 후 목록/")
    print("=" * 70)

    if torch.cuda.is_available():
        device = "cuda"; print("  [CUDA] NVIDIA GPU 감지")
    elif torch.backends.mps.is_available():
        device = "mps";  print("  [MPS] Apple Silicon 감지")
    else:
        device = "cpu";  print("  [CPU] GPU 없음")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 1024

    configs = build_embedding_configs(BASE_DIR)
    tasks   = [
        ("case_relevance_p.jsonl", "RELp_Case_to_Case"),
        ("case_relevance_q.jsonl", "RELq_Query_to_Case"),
    ]

    all_results = []
    for task_file, task_name in tasks:
        print(f"\n{'=' * 70}")
        print(f"  [{task_name}]")
        print(f"{'=' * 70}")
        for cfg_name, cfg_dict in configs.items():
            ngrams = cfg_dict.get("ngrams", [])
            words  = cfg_dict.get("words", set())
            print(f"  [{cfg_name}] n-gram {len(ngrams)}개, SW {len(words)}개 적용 중...", end=" ", flush=True)
            row = run_one(task_file, task_name, cfg_name, cfg_dict, model, device)
            all_results.append(row)
            print(f"R@1={row['R@1']}  R@5={row['R@5']}  R@10={row['R@10']}  MRR@10={row['MRR@10']}")

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("  최종 비교 결과")
    print("=" * 70)
    print(df.to_string(index=False))

    out_path = os.path.join(BASE_DIR, "06.result_embedding_precedents_검수후.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장: {out_path}")

if __name__ == "__main__":
    main()
