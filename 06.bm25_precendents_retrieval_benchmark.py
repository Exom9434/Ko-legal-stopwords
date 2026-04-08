import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# =============================================================================
# [Part 0] 설정 (Settings)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KBL_DIR  = os.path.join(BASE_DIR, "data", "kbl-rag")
TASKS_DIR = os.path.join(KBL_DIR, "tasks")
TOP_K_LIST = [1, 5, 10]

# =============================================================================
# [Part 1] 전처리 및 토크나이저 (기존 로직 유지)
# =============================================================================
def _init_mecab():
    try:
        from konlpy.tag import Mecab
        return Mecab(dicpath='/opt/homebrew/lib/mecab/dic/mecab-ko-dic')
    except Exception as e:
        return None

_mecab = _init_mecab()

def apply_ngram_stopwords(text: str, ngrams: list) -> str:
    if not ngrams: return text
    for phrase in ngrams:
        text = text.replace(phrase, " ")
    return text

def tokenize_nouns_filtered(text: str, config: dict) -> list[str]:
    if not text: return []
    text = apply_ngram_stopwords(text, config.get("ngrams", []))
    if _mecab:
        tokens = [word for word, pos in _mecab.pos(text) if pos.startswith("N")]
    else:
        tokens = [w for w in re.split(r"\s+", text) if len(w) >= 2]
    word_sw = config.get("words", set())
    return [t for t in tokens if t not in word_sw and len(t) >= 2]

# =============================================================================
# [Part 2] 불용어 로더 (기존 함수들 build_stopword_configs 포함하여 그대로 사용)
# =============================================================================
def load_stopwords_txt(path: str) -> set:
    if not os.path.exists(path): return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def load_stopwords_csv(path: str) -> set:
    if not os.path.exists(path): return set()
    df = pd.read_csv(path)
    return set(df["word"].astype(str).tolist()) if "word" in df.columns else set()

def _load_ngram_list(path: str) -> list:
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_stopword_configs(base_dir: str) -> dict:
    # 기초 불용어 로드
    kil = load_stopwords_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = load_stopwords_txt(os.path.join(base_dir, "stopwords.txt"))
    
    # 1. 판례 불용어 CSV 로드 (법령은 필요 시 로드만 하고 사용처 분리 가능)
    stat_dir_case = os.path.join(base_dir, "stopword_판례")
    case_a = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_A_zipf.csv"))
    case_b = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_B_entropy.csv"))
    case_c = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_C_mpvp.csv"))
    case_d = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_D_union.csv"))
    case_e = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_E_intersection.csv"))
    
    # 2. 판례 N-gram 로드
    ngram_case = _load_ngram_list(os.path.join(base_dir, "stopwords_ngram_판례.txt"))

    cfgs = {}

    # ==========================================
    # [그룹 0-1] 기초 및 N-gram 단독 효과 (00-07)
    # ==========================================
    cfgs["00. Baseline (No SW)"] = {"words": set(), "ngrams": []}
    cfgs["01. General (SW)"]     = {"words": gen, "ngrams": []}
    cfgs["02. KIL Only"]         = {"words": kil, "ngrams": []}
    cfgs["03. Baseline + 판례ngram-050"] = {"words": set(), "ngrams": ngram_case[:50]}
    cfgs["04. Baseline + 판례ngram-100"] = {"words": set(), "ngrams": ngram_case[:100]}
    cfgs["05. Baseline + 판례ngram-150"] = {"words": set(), "ngrams": ngram_case[:150]}
    cfgs["06. Baseline + 판례ngram-200"] = {"words": set(), "ngrams": ngram_case[:200]}
    cfgs["07. Baseline + 판례ngram-all"] = {"words": set(), "ngrams": ngram_case}

    # ==========================================
    # [그룹 2-3] 판례 불용어 단독 및 KIL 결합 (08-17)
    # ==========================================
    cases = {"A": case_a, "B": case_b, "C": case_c, "D": case_d, "E": case_e}
    for char, sw_set in cases.items():
        cfgs[f"08-{char}. Baseline + 판례{char}"] = {"words": sw_set, "ngrams": []}
    for char, sw_set in cases.items():
        cfgs[f"13-{char}. KIL + 판례{char}"]      = {"words": kil | sw_set, "ngrams": []}

    # ==========================================
    # [그룹 4-5] KIL/Baseline + N-gram 풀조합 (18-28)
    # ==========================================
    cfgs["18. KIL + 판례ngram"] = {"words": kil, "ngrams": ngram_case}
    for char, sw_set in cases.items():
        cfgs[f"19-{char}. Baseline + 판례{char} + 판례ngram"] = {"words": sw_set, "ngrams": ngram_case}
    for char, sw_set in cases.items():
        cfgs[f"24-{char}. KIL + 판례{char} + 판례ngram"]      = {"words": kil | sw_set, "ngrams": ngram_case}

    # ==========================================
    # [그룹 6-7] General SW 기반 대조군 (29-38)
    # ==========================================
    for char, sw_set in cases.items():
        cfgs[f"29-{char}. General + 판례{char}"] = {"words": gen | sw_set, "ngrams": []}
    for char, sw_set in cases.items():
        cfgs[f"34-{char}. General + 판례{char} + 판례ngram"] = {"words": gen | sw_set, "ngrams": ngram_case}

    # ==========================================
    # [그룹 8] KIL + N-gram 크기별 정밀 분석 (39-42)
    # ==========================================
    cfgs["39. KIL + 판례ngram-050"] = {"words": kil, "ngrams": ngram_case[:50]}
    cfgs["40. KIL + 판례ngram-100"] = {"words": kil, "ngrams": ngram_case[:100]}
    cfgs["41. KIL + 판례ngram-150"] = {"words": kil, "ngrams": ngram_case[:150]}
    cfgs["42. KIL + 판례ngram-200"] = {"words": kil, "ngrams": ngram_case[:200]}

    # ==========================================
    # [그룹 9] Baseline + 판례E + N-gram 크기별 (43-46)
    # ==========================================
    cfgs["43. Baseline + 판례E + 판례ngram-050"] = {"words": case_e, "ngrams": ngram_case[:50]}
    cfgs["44. Baseline + 판례E + 판례ngram-100"] = {"words": case_e, "ngrams": ngram_case[:100]}
    cfgs["45. Baseline + 판례E + 판례ngram-150"] = {"words": case_e, "ngrams": ngram_case[:150]}
    cfgs["46. Baseline + 판례E + 판례ngram-200"] = {"words": case_e, "ngrams": ngram_case[:200]}

    # ==========================================
    # [그룹 10] General + N-gram 크기별 (47-50)
    # ==========================================
    cfgs["47. General + 판례ngram-050"] = {"words": gen, "ngrams": ngram_case[:50]}
    cfgs["48. General + 판례ngram-100"] = {"words": gen, "ngrams": ngram_case[:100]}
    cfgs["49. General + 판례ngram-150"] = {"words": gen, "ngrams": ngram_case[:150]}
    cfgs["50. General + 판례ngram-200"] = {"words": gen, "ngrams": ngram_case[:200]}

    return cfgs
# =============================================================================
# [Part 3] Case Relevance 전용 로더 (중요 수정 포인트)
# =============================================================================

def load_case_relevance_task(task_file: str):
    """
    RELp 또는 RELq jsonl 파일을 읽어서 
    1. 검색 대상이 될 문서 풀(Corpus)
    2. 검색어가 될 쿼리(Query)와 정답 ID 매핑
    을 생성합니다.
    """
    path = os.path.join(TASKS_DIR, task_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")

    queries = []
    corpus_dict = {} # 중복 방지를 위해 dict 사용 (text -> doc_id)
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q_text = item["query"]
            r_text = item["retrieved_text"]
            is_relevant = item["is_relevant"]
            
            # 1. 문서 풀 구축 (retrieved_text를 고유 문서로 취급)
            if r_text not in corpus_dict:
                doc_id = f"doc_{len(corpus_dict)}"
                corpus_dict[r_text] = doc_id
            
            target_doc_id = corpus_dict[r_text]
            
            # 2. 쿼리 구성 (정답인 경우만 gold_ids에 추가)
            # 1:1 매칭 구조이므로 gold_ids는 1개거나 0개(Negative 쌍)가 됨
            gold_ids = [target_doc_id] if is_relevant else []
            
            queries.append({
                "query": q_text,
                "gold_ids": gold_ids
            })

    # 최종 코퍼스 리스트 변환
    corpus = [{"id": vid, "text": txt} for txt, vid in corpus_dict.items()]
    return corpus, queries

# =============================================================================
# [Part 4] 평가 로직 (기존 BM25Index 및 evaluate 활용)
# =============================================================================

# (기존의 BM25Index 클래스, recall_at_k, reciprocal_rank, evaluate 함수 그대로 사용)
class BM25Index:
    def __init__(self, corpus_ids: list, base_tokens_list: list, config: dict):
        self.ids = corpus_ids
        word_sw = config.get("words", set())
        
        # 🚨 여기서 MeCab은 돌지 않습니다! 
        # 미리 쪼개둔 형태소 리스트에서 불용어(word_sw)만 빛의 속도로 걸러냅니다.
        final_tokenized = [
            [t for t in tokens if t not in word_sw]
            for tokens in base_tokens_list
        ]
        
        # 필터링된 최종 토큰으로 BM25 색인 (수 초 내로 끝남)
        self.bm25 = BM25Okapi(final_tokenized)

    def search(self, query: str, config: dict, top_k=10):
        # 질문(Query)은 1문장이니 실시간으로 쪼개도 순식간입니다.
        tokens = tokenize_nouns_filtered(query, config)
        if not tokens: return []
        scores = self.bm25.get_scores(tokens)
        return [self.ids[i] for i in np.argsort(scores)[::-1][:top_k]]

def recall_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    if not gold_ids: return 0.0
    gold_set = set(gold_ids)
    return 1.0 if any(rid in gold_set for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(retrieved_ids: list, gold_ids: list) -> float:
    if not gold_ids: return 0.0
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set: return 1.0 / rank
    return 0.0

def evaluate(queries, corpus_ids, base_tokens_list, config):
    # 이제 캐싱된 토큰 리스트를 그대로 넘깁니다.
    index = BM25Index(corpus_ids, base_tokens_list, config)

    recalls = {k: [] for k in TOP_K_LIST}
    rrs = []

    for q in queries: # 쿼리 검색에는 tqdm 뺐습니다 (너무 빨리 지나가서 번쩍임 방지)
        gold_ids = q.get("gold_ids", [])
        if not gold_ids: continue # 정답이 없는 쿼리는 패스
        
        res = index.search(q["query"], config, top_k=max(TOP_K_LIST))

        for k in TOP_K_LIST:
            recalls[k].append(recall_at_k(res, gold_ids, k))
        rrs.append(reciprocal_rank(res, gold_ids))
        
    return {f"Recall@{k}": np.mean(recalls[k]) for k in TOP_K_LIST} | {"MRR@10": np.mean(rrs)}
# =============================================================================
# [Part 5] 메인 실행 (RELp, RELq 각각 루프)
# =============================================================================

def run_benchmark(task_file, task_display_name, configs):
    print(f"\n\n🚀 [{task_display_name}] 벤치마크 시작")
    corpus, queries = load_case_relevance_task(task_file)
    
    # 1. 코퍼스 캐싱
    corpus_ids = [doc["id"] for doc in corpus]
    corpus_cache = {}
    unique_ngrams = set(tuple(cfg.get("ngrams", [])) for cfg in configs.values())
    
    for ng_tuple in unique_ngrams:
        temp_config = {"words": set(), "ngrams": list(ng_tuple)}
        base_tokens_list = [tokenize_nouns_filtered(doc["text"], temp_config) for doc in corpus]
        corpus_cache[ng_tuple] = base_tokens_list

    # 2. 실험 루프
    results = []
    print("-" * 75)
    print(f"{'실험 설정':<30} | R@1    R@5    MRR@10")
    print("-" * 75)
    
    for name, cfg in configs.items():
        ng_tuple = tuple(cfg.get("ngrams", []))
        base_tokens = corpus_cache[ng_tuple]
        
        m = evaluate(queries, corpus_ids, base_tokens, cfg)
        print(f"{name:<30} | {m['Recall@1']:.4f} {m['Recall@5']:.4f} {m['MRR@10']:.4f}")
        results.append({"Experiment": name} | m)
        
    # 결과 저장
    df = pd.DataFrame(results).sort_values("MRR@10", ascending=False)
    df.to_csv(os.path.join(BASE_DIR, f"06.result_bm25_{task_display_name}.csv"), index=False, encoding="utf-8-sig")

def main():
    # 1. 설정 로드
    #from stopword_configs import build_stopword_configs # 설정 함수를 분리했을 경우
    configs = build_stopword_configs(BASE_DIR)
    
    # 2. RELp (판례 vs 판례) 벤치마크 실행
    run_benchmark("case_relevance_p.jsonl", "RELp_Case_to_Case", configs)
    
    # 3. RELq (질의 vs 판례) 벤치마크 실행
    run_benchmark("case_relevance_q.jsonl", "RELq_Query_to_Case", configs)

if __name__ == "__main__":
    main()