import os, json, re, torch, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KBL_DIR    = os.path.join(BASE_DIR, "data", "kbl-rag")
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
# 변경점: stopword_법령/candidates_E_intersection.csv → 검수 후 목록/stopwords_단일어_법령_E_intersection_final.txt
#         stopwords_ngram_법령.txt                    → 검수 후 목록/stopwords_ngram_법령_final.txt
# =============================================================================
def build_embedding_configs(base_dir):
    reviewed = os.path.join(base_dir, "검수 후 목록")
    law_e    = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_법령_E_intersection_final.txt"))
    ngram_law = _load_ngram_list(os.path.join(reviewed, "stopwords_ngram_법령_final.txt"))

    return {
        "00_Baseline":          {"ngrams": [],        "words": set()},
        "01_법령ngram_검수후":   {"ngrams": ngram_law, "words": set()},
        "02_법령E_검수후":       {"ngrams": [],        "words": law_e},
        "03_법령E+법령ngram_검수후": {"ngrams": ngram_law, "words": law_e},
    }

def load_kbl_corpus(kbl_dir):
    path = os.path.join(kbl_dir, "corpus", "statutes.jsonl")
    if not os.path.exists(path): path = os.path.join(kbl_dir, "corpus", "statutes_local.jsonl")
    if not os.path.exists(path): raise FileNotFoundError(f"코퍼스 없음: {path}")
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj.get("id", ""))
            text   = obj.get("contents") or obj.get("text", "")
            m = re.search(r"docid-(\d+)__", doc_id)
            corpus.append({"id": doc_id, "text": text, "docid_num": int(m.group(1)) if m else -1})
    return corpus

MANUAL_GOLD_MAP = {
    1:  ["docid-8590__dumpdate-20231128___jo-50"],
    2:  ["docid-9874__dumpdate-20231128___jo-23"],
    3:  ["docid-3191__dumpdate-20231128___jo-5"],
    4:  ["docid-11476__dumpdate-20231128___jo-385"],
    5:  ["docid-11476__dumpdate-20231128___jo-271"],
    6:  ["docid-9908__dumpdate-20231128___jo-98"],
    7:  ["docid-10809__dumpdate-20231128___jo-248"],
    8:  ["docid-11476__dumpdate-20231128___jo-323"],
    9:  ["docid-5319__dumpdate-20231128___jo-1"],
    10: ["docid-10809__dumpdate-20231128___jo-392"],
    11: ["docid-11476__dumpdate-20231128___jo-52"],
    12: ["docid-10305__dumpdate-20231128___jo-52"],
    13: ["docid-9882__dumpdate-20231128___jo-28"],
    14: ["docid-10757__dumpdate-20231128___jo-8"],
    15: ["docid-11476__dumpdate-20231128___jo-240"],
    16: ["docid-3044__dumpdate-20231128___jo-43"],
    17: ["docid-11377__dumpdate-20231128___jo-7"],
    18: ["docid-12010__dumpdate-20231128___jo-183"],
    19: ["docid-12010__dumpdate-20231128___jo-183"],
    20: ["docid-10022__dumpdate-20231128___jo-155"],
    21: ["docid-5653__dumpdate-20231128___jo-468"],
    22: ["docid-11476__dumpdate-20231128___jo-267"],
    23: ["docid-11022__dumpdate-20231128___jo-22"],
    24: ["docid-5653__dumpdate-20231128___jo-308"],
    25: ["docid-4001__dumpdate-20231128___jo-35"],
    26: ["docid-11255__dumpdate-20231128___jo-41"],
    27: ["docid-10809__dumpdate-20231128___jo-1078"],
    28: ["docid-10809__dumpdate-20231128___jo-113"],
    29: ["docid-9908__dumpdate-20231128___jo-469"],
    30: ["docid-11476__dumpdate-20231128___jo-51"],
    31: ["docid-6020__dumpdate-20231128___jo-58"],
    32: ["docid-1516__dumpdate-20231128___jo-20"],
    33: ["docid-10809__dumpdate-20231128___jo-325"],
    34: ["docid-11476__dumpdate-20231128___jo-382"],
    35: ["docid-5653__dumpdate-20231128___jo-473"],
    36: ["docid-10791__dumpdate-20231128___jo-40"],
    37: ["docid-9962__dumpdate-20231128___jo-188"],
    38: ["docid-11476__dumpdate-20231128___jo-36"],
    39: ["docid-11476__dumpdate-20231128___jo-314"],
    40: ["docid-4001__dumpdate-20231128___jo-118"],
    41: ["docid-11077__dumpdate-20231128___jo-92"],
    42: ["docid-4217__dumpdate-20231128___jo-63"],
    43: ["docid-11476__dumpdate-20231128___jo-220"],
    44: ["docid-5310__dumpdate-20231128___jo-3"],
    45: ["docid-11476__dumpdate-20231128___jo-34"],
    46: ["docid-11476__dumpdate-20231128___jo-309"],
    47: ["docid-11476__dumpdate-20231128___jo-62"],
    48: ["docid-12010__dumpdate-20231128___jo-56"],
    49: ["docid-4001__dumpdate-20231128___jo-22"],
    50: ["docid-5653__dumpdate-20231128___jo-438"],
    51: ["docid-4001__dumpdate-20231128___jo-119"],
    52: ["docid-4001__dumpdate-20231128___jo-123"],
}

def load_kbl_queries(kbl_dir, task):
    path = os.path.join(kbl_dir, "tasks", f"{task}.jsonl")
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question") or ""
            if not q: continue
            doc_id   = obj.get("doc_id")
            gold_ids = MANUAL_GOLD_MAP.get(int(doc_id), []) if doc_id is not None else []
            queries.append({"query": q, "gold_ids": gold_ids})
    return queries

def recall_at_k(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    return 1.0 if any(rid in gold_set for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set: return 1.0 / rank
    return 0.0

def encode_corpus(corpus_texts, config, cfg_name, model, device):
    # 검수후 버전은 캐시 파일명에 _검수후 접미사를 붙여 기존 캐시와 분리
    cache_path = os.path.join(BASE_DIR, f"corpus_emb_statute_{cfg_name}_검수후.pt")

    if os.path.exists(cache_path):
        print(f"  캐시 로드: {os.path.basename(cache_path)}")
        return torch.load(cache_path, map_location=device)

    ngrams = config.get("ngrams", [])
    words  = config.get("words", set())

    print(f"  [{cfg_name}] 코퍼스 전처리 및 인코딩 시작 ({len(corpus_texts):,}개)...")
    preprocessed = [apply_clean_text(t, ngrams, words) for t in corpus_texts]

    all_embs = []
    chunk_size = 10000
    for i in range(0, len(preprocessed), chunk_size):
        chunk = preprocessed[i : i + chunk_size]
        emb = model.encode(chunk, batch_size=16, show_progress_bar=True,
                           convert_to_tensor=True).cpu()
        all_embs.append(emb)
        if device == "cuda":  torch.cuda.empty_cache()
        elif device == "mps": torch.mps.empty_cache()
        gc.collect()

    corpus_embeddings = torch.cat(all_embs, dim=0)
    torch.save(corpus_embeddings, cache_path)
    return corpus_embeddings.to(device)

def main():
    print("=" * 70)
    print(f"  Embedding Retrieval Benchmark (법령) — 검수 후 버전")
    print(f"  모델: {MODEL_NAME}")
    print(f"  불용어 경로: 검수 후 목록/")
    print("=" * 70)

    print("\n[1] 데이터 로드 중...")
    corpus       = load_kbl_corpus(KBL_DIR)
    queries      = load_kbl_queries(KBL_DIR, "query_statute_matching")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids   = [doc["id"]   for doc in corpus]
    valid_queries = [q for q in queries if q["gold_ids"]]
    print(f"  corpus {len(corpus_texts):,}개 / 유효 쿼리 {len(valid_queries)}개")

    print(f"\n[2] 모델 로드 중... ({MODEL_NAME})")
    if torch.cuda.is_available():
        device = "cuda"; print("  [CUDA] NVIDIA GPU 감지")
    elif torch.backends.mps.is_available():
        device = "mps";  print("  [MPS] Apple Silicon 감지")
    else:
        device = "cpu";  print("  [CPU] GPU 없음")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 1024

    configs = build_embedding_configs(BASE_DIR)
    all_results = []

    for cfg_name, cfg_dict in configs.items():
        ngrams = cfg_dict.get("ngrams", [])
        words  = cfg_dict.get("words", set())
        print(f"\n--- [{cfg_name}] (n-gram:{len(ngrams)}개, SW:{len(words)}개) ---")

        corpus_embeddings = encode_corpus(corpus_texts, cfg_dict, cfg_name, model, device)

        query_texts = [apply_clean_text(q["query"], ngrams, words) for q in valid_queries]
        query_embeddings = model.encode(
            query_texts, batch_size=32, show_progress_bar=False, convert_to_tensor=True
        )

        all_hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=max(TOP_K_LIST))
        recalls = {k: [] for k in TOP_K_LIST}
        rrs = []
        for q, hits in zip(valid_queries, all_hits):
            retrieved_ids = [corpus_ids[hit["corpus_id"]] for hit in hits]
            for k in TOP_K_LIST:
                recalls[k].append(recall_at_k(retrieved_ids, q["gold_ids"], k))
            rrs.append(reciprocal_rank(retrieved_ids, q["gold_ids"]))

        row = {"설정": cfg_name,
               **{f"R@{k}": round(np.mean(recalls[k]), 4) for k in TOP_K_LIST},
               "MRR@10": round(np.mean(rrs), 4)}
        all_results.append(row)
        print(f"  R@1={row['R@1']}  R@5={row['R@5']}  R@10={row['R@10']}  MRR@10={row['MRR@10']}")

        del corpus_embeddings, query_embeddings
        if device == "cuda":  torch.cuda.empty_cache()
        elif device == "mps": torch.mps.empty_cache()
        gc.collect()

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("  최종 비교 결과")
    print("=" * 70)
    print(df.to_string(index=False))

    out_path = os.path.join(BASE_DIR, "06.result_embedding_statue_검수후.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장: {out_path}")

if __name__ == "__main__":
    main()
