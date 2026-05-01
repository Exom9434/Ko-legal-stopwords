import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
KBL_DIR   = os.path.join(BASE_DIR, "data", "kbl-rag")
TASKS_DIR = os.path.join(KBL_DIR, "tasks")
TOP_K_LIST = [1, 5, 10]

def _init_mecab():
    try:
        from konlpy.tag import Mecab
        return Mecab(dicpath='/opt/homebrew/lib/mecab/dic/mecab-ko-dic')
    except Exception as e:
        return None

_mecab = _init_mecab()

def apply_ngram_stopwords(text, ngrams):
    if not ngrams: return text
    for phrase in ngrams:
        text = text.replace(phrase, " ")
    return text

def tokenize_nouns_filtered(text, config):
    if not text: return []
    text = apply_ngram_stopwords(text, config.get("ngrams", []))
    if _mecab:
        tokens = [word for word, pos in _mecab.pos(text) if pos.startswith("N")]
    else:
        tokens = [w for w in re.split(r"\s+", text) if len(w) >= 2]
    word_sw = config.get("words", set())
    return [t for t in tokens if t not in word_sw and len(t) >= 2]

def load_stopwords_txt(path):
    if not os.path.exists(path): return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def _load_ngram_list(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# =============================================================================
# build_stopword_configs — 검수 후 버전
# 변경점: stopword_판례/candidates_*.csv → 검수 후 목록/stopwords_단일어_판례_*_final.txt
#         stopwords_ngram_판례.txt       → 검수 후 목록/stopwords_ngram_판례_final.txt
# =============================================================================
def build_stopword_configs(base_dir):
    kil = load_stopwords_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = load_stopwords_txt(os.path.join(base_dir, "stopwords.txt"))

    reviewed = os.path.join(base_dir, "검수 후 목록")
    case_a = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_A_zipf_final.txt"))
    case_b = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_B_entropy_final.txt"))
    case_c = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_C_mpvp_final.txt"))
    case_d = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_D_union_final.txt"))
    case_e = load_stopwords_txt(os.path.join(reviewed, "stopwords_단일어_판례_E_intersection_final.txt"))
    ngram_case = _load_ngram_list(os.path.join(reviewed, "stopwords_ngram_판례_final.txt"))

    cfgs = {}
    cfgs["00. Baseline (No SW)"] = {"words": set(), "ngrams": []}
    cfgs["01. General (SW)"]     = {"words": gen, "ngrams": []}
    cfgs["02. KIL Only"]         = {"words": kil, "ngrams": []}
    cfgs["03. Baseline + 판례ngram-050"] = {"words": set(), "ngrams": ngram_case[:50]}
    cfgs["04. Baseline + 판례ngram-100"] = {"words": set(), "ngrams": ngram_case[:100]}
    cfgs["05. Baseline + 판례ngram-150"] = {"words": set(), "ngrams": ngram_case[:150]}
    cfgs["06. Baseline + 판례ngram-200"] = {"words": set(), "ngrams": ngram_case[:200]}
    cfgs["07. Baseline + 판례ngram-all"] = {"words": set(), "ngrams": ngram_case}

    cases = {"A": case_a, "B": case_b, "C": case_c, "D": case_d, "E": case_e}
    for char, sw_set in cases.items():
        cfgs[f"08-{char}. Baseline + 판례{char}"] = {"words": sw_set, "ngrams": []}
    for char, sw_set in cases.items():
        cfgs[f"13-{char}. KIL + 판례{char}"]      = {"words": kil | sw_set, "ngrams": []}

    cfgs["18. KIL + 판례ngram"] = {"words": kil, "ngrams": ngram_case}
    for char, sw_set in cases.items():
        cfgs[f"19-{char}. Baseline + 판례{char} + 판례ngram"] = {"words": sw_set, "ngrams": ngram_case}
    for char, sw_set in cases.items():
        cfgs[f"24-{char}. KIL + 판례{char} + 판례ngram"]      = {"words": kil | sw_set, "ngrams": ngram_case}

    for char, sw_set in cases.items():
        cfgs[f"29-{char}. General + 판례{char}"] = {"words": gen | sw_set, "ngrams": []}
    for char, sw_set in cases.items():
        cfgs[f"34-{char}. General + 판례{char} + 판례ngram"] = {"words": gen | sw_set, "ngrams": ngram_case}

    cfgs["39. KIL + 판례ngram-050"] = {"words": kil, "ngrams": ngram_case[:50]}
    cfgs["40. KIL + 판례ngram-100"] = {"words": kil, "ngrams": ngram_case[:100]}
    cfgs["41. KIL + 판례ngram-150"] = {"words": kil, "ngrams": ngram_case[:150]}
    cfgs["42. KIL + 판례ngram-200"] = {"words": kil, "ngrams": ngram_case[:200]}

    cfgs["43. Baseline + 판례E + 판례ngram-050"] = {"words": case_e, "ngrams": ngram_case[:50]}
    cfgs["44. Baseline + 판례E + 판례ngram-100"] = {"words": case_e, "ngrams": ngram_case[:100]}
    cfgs["45. Baseline + 판례E + 판례ngram-150"] = {"words": case_e, "ngrams": ngram_case[:150]}
    cfgs["46. Baseline + 판례E + 판례ngram-200"] = {"words": case_e, "ngrams": ngram_case[:200]}

    cfgs["47. General + 판례ngram-050"] = {"words": gen, "ngrams": ngram_case[:50]}
    cfgs["48. General + 판례ngram-100"] = {"words": gen, "ngrams": ngram_case[:100]}
    cfgs["49. General + 판례ngram-150"] = {"words": gen, "ngrams": ngram_case[:150]}
    cfgs["50. General + 판례ngram-200"] = {"words": gen, "ngrams": ngram_case[:200]}

    return cfgs

def load_case_relevance_task(task_file):
    path = os.path.join(TASKS_DIR, task_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")
    queries = []
    corpus_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q_text = item["query"]
            r_text = item["retrieved_text"]
            is_relevant = item["is_relevant"]
            if r_text not in corpus_dict:
                doc_id = f"doc_{len(corpus_dict)}"
                corpus_dict[r_text] = doc_id
            target_doc_id = corpus_dict[r_text]
            gold_ids = [target_doc_id] if is_relevant else []
            queries.append({"query": q_text, "gold_ids": gold_ids})
    corpus = [{"id": vid, "text": txt} for txt, vid in corpus_dict.items()]
    return corpus, queries

class BM25Index:
    def __init__(self, corpus_ids, base_tokens_list, config):
        self.ids = corpus_ids
        word_sw = config.get("words", set())
        final_tokenized = [[t for t in tokens if t not in word_sw] for tokens in base_tokens_list]
        self.bm25 = BM25Okapi(final_tokenized)
    def search(self, query, config, top_k=10):
        tokens = tokenize_nouns_filtered(query, config)
        if not tokens: return []
        scores = self.bm25.get_scores(tokens)
        return [self.ids[i] for i in np.argsort(scores)[::-1][:top_k]]

def recall_at_k(retrieved_ids, gold_ids, k):
    if not gold_ids: return 0.0
    return 1.0 if any(rid in set(gold_ids) for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(retrieved_ids, gold_ids):
    if not gold_ids: return 0.0
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set: return 1.0 / rank
    return 0.0

def evaluate(queries, corpus_ids, base_tokens_list, config):
    index = BM25Index(corpus_ids, base_tokens_list, config)
    recalls = {k: [] for k in TOP_K_LIST}
    rrs = []
    for q in queries:
        gold_ids = q.get("gold_ids", [])
        if not gold_ids: continue
        res = index.search(q["query"], config, top_k=max(TOP_K_LIST))
        for k in TOP_K_LIST:
            recalls[k].append(recall_at_k(res, gold_ids, k))
        rrs.append(reciprocal_rank(res, gold_ids))
    return {f"Recall@{k}": np.mean(recalls[k]) for k in TOP_K_LIST} | {"MRR@10": np.mean(rrs)}

def run_benchmark(task_file, task_display_name, configs):
    print(f"\n\n🚀 [{task_display_name}] 벤치마크 시작")
    corpus, queries = load_case_relevance_task(task_file)
    corpus_ids = [doc["id"] for doc in corpus]
    corpus_cache = {}
    unique_ngrams = set(tuple(cfg.get("ngrams", [])) for cfg in configs.values())
    for ng_tuple in unique_ngrams:
        temp_config = {"words": set(), "ngrams": list(ng_tuple)}
        corpus_cache[ng_tuple] = [tokenize_nouns_filtered(doc["text"], temp_config) for doc in corpus]

    results = []
    print("-" * 75)
    print(f"{'실험 설정':<30} | R@1    R@5    MRR@10")
    print("-" * 75)
    for name, cfg in configs.items():
        ng_tuple = tuple(cfg.get("ngrams", []))
        m = evaluate(queries, corpus_ids, corpus_cache[ng_tuple], cfg)
        print(f"{name:<30} | {m['Recall@1']:.4f} {m['Recall@5']:.4f} {m['MRR@10']:.4f}")
        results.append({"Experiment": name} | m)

    df = pd.DataFrame(results).sort_values("MRR@10", ascending=False)
    df.to_csv(os.path.join(BASE_DIR, f"06.result_bm25_{task_display_name}_검수후.csv"),
              index=False, encoding="utf-8-sig")

def main():
    print("=" * 65)
    print(" KBL BM25 Retrieval Benchmark — 검수 후 버전 (판례)")
    print(f" 불용어 경로: 검수 후 목록/")
    print("=" * 65)
    configs = build_stopword_configs(BASE_DIR)
    run_benchmark("case_relevance_p.jsonl", "RELp_Case_to_Case_검수후", configs)
    run_benchmark("case_relevance_q.jsonl", "RELq_Query_to_Case_검수후", configs)

if __name__ == "__main__":
    main()
