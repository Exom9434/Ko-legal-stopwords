import os
import re
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# =============================================================================
# [Part 0] 설정 (Settings)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KBL_DIR  = os.path.join(BASE_DIR, "data", "kbl-rag")
USE_DUMMY_DATA = not os.path.exists(KBL_DIR)
TOP_K_LIST    = [1, 5, 10]

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
    19: ["docid-9908__dumpdate-20231128___jo-212"],  # ← 18번이랑 동일, 맞는지 확인 필요
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

# =============================================================================
# [Part 1] 전처리 및 토크나이저 (n-gram 필터 포함)
# =============================================================================
def _init_mecab():
    try:
        from konlpy.tag import Mecab
        # 방금 성공했던 시스템 사전 경로를 고정으로 박아줍니다!
        return Mecab(dicpath='/opt/homebrew/lib/mecab/dic/mecab-ko-dic')
    except Exception as e:
        print(f"  [경고] MeCab 초기화 실패 ({e}) → 어절 기반 폴백 사용")
        return None

_mecab = _init_mecab()

def apply_ngram_stopwords(text: str, ngrams: list) -> str:
    """원문에서 n-gram 구문 삭제"""
    if not ngrams: return text
    for phrase in ngrams:
        text = text.replace(phrase, " ")
    return text

def tokenize_nouns_filtered(text: str, config: dict) -> list[str]:
    """n-gram 제거 -> 명사 추출 -> 불용어 제거"""
    if not text: return []
    
    # 1. n-gram 제거 (구절 단위)
    text = apply_ngram_stopwords(text, config.get("ngrams", []))
    
    # 2. 명사 추출
    if _mecab:
        tokens = [word for word, pos in _mecab.pos(text) if pos.startswith("N")]
    else:
        tokens = [w for w in re.split(r"\s+", text) if len(w) >= 2 and re.search(r"[가-힣]", w)]
    
    # 3. 단어 불용어 제거
    word_sw = config.get("words", set())
    return [t for t in tokens if t not in word_sw and len(t) >= 2]

# =============================================================================
# [Part 2] 불용어 & n-gram 로더
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

# =============================================================================
# [Part 2] 불용어 & n-gram 로더 (17개 설정 풀버전)
# =============================================================================
def build_stopword_configs(base_dir: str) -> dict:
    kil = load_stopwords_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = load_stopwords_txt(os.path.join(base_dir, "stopwords.txt"))
    
    # 1. 법령 불용어 CSV 로드
    stat_dir_law = os.path.join(base_dir, "stopword_법령")
    law_a = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_A_zipf.csv"))
    law_b = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_B_tfidf.csv"))
    law_c = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_C_entropy.csv"))
    law_d = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_D_pos.csv"))
    law_e = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_E_intersection.csv"))
    
    # 2. 판례 불용어 CSV 로드
    stat_dir_case = os.path.join(base_dir, "stopword_판례")
    case_a = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_A_zipf.csv"))
    case_b = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_B_tfidf.csv"))
    case_c = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_C_entropy.csv"))
    case_d = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_D_pos.csv"))
    case_e = load_stopwords_csv(os.path.join(stat_dir_case, "candidates_E_intersection.csv"))
    
    # 3. N-gram 로드
    ngram_law = _load_ngram_list(os.path.join(base_dir, "stopwords_ngram_법령.txt"))
    ngram_case = _load_ngram_list(os.path.join(base_dir, "stopwords_ngram_판례.txt"))

    return {
        "00. Baseline":        {"words": set(), "ngrams": []},
        "01. General (SW)":    {"words": gen, "ngrams": []},
        "02. KIL Only":        {"words": kil, "ngrams": []},
        
        # 법령 단어 불용어 A~E 시리즈
        "03. KIL + 법령A":       {"words": kil | law_a, "ngrams": []},
        "04. KIL + 법령B":       {"words": kil | law_b, "ngrams": []},
        "05. KIL + 법령C":       {"words": kil | law_c, "ngrams": []},
        "06. KIL + 법령D":       {"words": kil | law_d, "ngrams": []},
        "07. KIL + 법령E":       {"words": kil | law_e, "ngrams": []},
        
        # 판례 단어 불용어 A~E 시리즈
        "08. KIL + 판례A":       {"words": kil | case_a, "ngrams": []},
        "09. KIL + 판례B":       {"words": kil | case_b, "ngrams": []},
        "10. KIL + 판례C":       {"words": kil | case_c, "ngrams": []},
        "11. KIL + 판례D":       {"words": kil | case_d, "ngrams": []},
        "12. KIL + 판례E":       {"words": kil | case_e, "ngrams": []},
        
        # N-gram 시리즈
        "13. KIL + 법령ngram":   {"words": kil, "ngrams": ngram_law},
        "14. KIL + 판례ngram":   {"words": kil, "ngrams": ngram_case},
        
        # 궁극의 결합 테스트
        "15. KIL + 법령E + ngram": {"words": kil | law_e, "ngrams": ngram_law},
        "16. KIL + 판례E + ngram": {"words": kil | case_e, "ngrams": ngram_case},
    }
# =============================================================================
# [Part 3] KBL 데이터 로더 (중요: 에러 났던 함수들)
# =============================================================================
def load_kbl_corpus(kbl_dir: str) -> list[dict]:
    path = os.path.join(kbl_dir, "corpus", "statutes.jsonl")
    if not os.path.exists(path): path = os.path.join(kbl_dir, "corpus", "statutes_local.jsonl")
    if not os.path.exists(path): raise FileNotFoundError(f"코퍼스 없음: {path}")

    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj.get("id", ""))
            text = obj.get("contents") or obj.get("text", "")
            m = re.search(r"docid-(\d+)__", doc_id)
            corpus.append({"id": doc_id, "text": text, "docid_num": int(m.group(1)) if m else -1})
    return corpus

def build_docid_index(corpus: list[dict]) -> dict:
    """
    corpus ID ("docid-N__dumpdate-...___jo-M") 에서 N을 추출해
    {N: [corpus_id, ...]} 역인덱스 구축.
    한 법령(docid-N)에 여러 조문(jo-0, jo-1, ...)이 존재할 수 있으므로 리스트로 관리.
    """
    index = {}
    for doc in corpus:
        m = re.search(r"docid-(\d+)__", doc["id"])
        if m:
            n = int(m.group(1))
            index.setdefault(n, []).append(doc["id"])
    return index

def load_kbl_queries(kbl_dir, task, corpus):
    path = os.path.join(kbl_dir, "tasks", f"{task}.jsonl")
    if not os.path.exists(path): return []

    print("  [알림] Bi-gram 기반 '포함도(Containment)' 정답 매핑을 시작합니다...")
    
    def get_char_bigrams(text):
        clean = re.sub(r'[^가-힣0-9]', '', text)
        return set(clean[i:i+2] for i in range(len(clean)-1))

    # 1. 코퍼스 Bi-gram 인덱싱
    norm_corpus = []
    for doc in tqdm(corpus, desc="  코퍼스 Bi-gram 인덱싱 중", leave=False):
        norm_corpus.append((doc["id"], get_char_bigrams(doc["text"])))

    queries = []
    matched, unmatched = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question") or ""
            if not q: continue

            doc_id = obj.get("doc_id")
            gold_ids = MANUAL_GOLD_MAP.get(int(doc_id), []) if doc_id is not None else []

            queries.append({"query": q, "gold_ids": gold_ids})

    print(f"  [gold 매핑] 성공 {matched}건 / 실패 {unmatched}건")
    return queries

def load_dummy_data():
    print("  [더미 모드] KBL 대신 프로젝트 내부 데이터 사용")
    corpus = [{"id": f"doc_{i}", "text": f"법령 내용 샘플 {i}"} for i in range(100)]
    queries = [{"query": "샘플 쿼리", "gold_ids": ["doc_0"]}]
    return corpus, queries

# =============================================================================
# [Part 4] BM25 엔진 및 평가 (초고속 캐싱 버전)
# =============================================================================
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
# [Part 5] 메인 실행 (초고속 캐싱 로직 포함)
# =============================================================================
def main():
    print("=" * 65)
    print(" KBL BM25 Retrieval Benchmark — 초고속 캐싱 통합 버전")
    print("=" * 65)

    if USE_DUMMY_DATA:
        corpus, queries = load_dummy_data()
        task_name = "dummy"
    else:
        task_name = "query_statute_matching"
        corpus = load_kbl_corpus(KBL_DIR)
        queries = load_kbl_queries(KBL_DIR, task_name, corpus) # Jaccard 매핑 함수 적용됨

    configs = build_stopword_configs(BASE_DIR)
    results = []
    
    if len(queries) == 0:
        print("❌ 에러: 쿼리가 하나도 로드되지 않았습니다!")
        return

    # -------------------------------------------------------------------------
    # 🔥 초고속 색인을 위한 코퍼스 사전 캐싱 (Caching)
    # -------------------------------------------------------------------------
    print("\n[알림] 🚀 초고속 색인을 위해 코퍼스를 메모리에 캐싱합니다...")
    corpus_ids = [doc["id"] for doc in corpus]
    corpus_cache = {}
    
    # 설정들 중에서 고유한 n-gram 리스트만 뽑아냄 (tuple로 변환하여 dict key로 사용)
    unique_ngrams = set(tuple(cfg.get("ngrams", [])) for cfg in configs.values())
    
    for ng_tuple in unique_ngrams:
        print(f"  - n-gram 캐싱 중... (설정된 n-gram 수: {len(ng_tuple)}개)")
        temp_config = {"words": set(), "ngrams": list(ng_tuple)}
        
        base_tokens_list = []
        for doc in tqdm(corpus, desc="    MeCab 형태소 분석", leave=False):
            # n-gram을 지우고 MeCab으로 명사만 쪼갠 상태를 저장합니다.
            base_tokens_list.append(tokenize_nouns_filtered(doc["text"], temp_config))
            
        # 완성된 리스트를 메모리에 저장
        corpus_cache[ng_tuple] = base_tokens_list
        
    print("  ✅ 캐싱 완료! 이제 16개의 실험이 빛의 속도로 진행됩니다.\n")
    # -------------------------------------------------------------------------

    print("-" * 75)
    print(f"{'실험 설정':<25} | R@1    R@5    R@10   MRR@10")
    print("-" * 75)

    for name, cfg in configs.items():
        # 현재 실험의 n-gram 설정을 키(Key)로 변환
        ng_tuple = tuple(cfg.get("ngrams", []))
        
        # 이미 쪼개져 있는 형태소 리스트를 캐시에서 1초 만에 꺼내옴!
        base_tokens = corpus_cache[ng_tuple] 
        
        # 평가 함수에 넘김
        m = evaluate(queries, corpus_ids, base_tokens, cfg)
        print(f"{name:<25} | {m['Recall@1']:.4f} {m['Recall@5']:.4f} {m['Recall@10']:.4f} {m['MRR@10']:.4f}")
        results.append({"Experiment": name} | m)

    df = pd.DataFrame(results).sort_values("MRR@10", ascending=False)
    df.to_csv(os.path.join(BASE_DIR, f"06.result_bm25_{task_name}.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    print("-" * 75)
    print(f"\n✅ 완료! 결과가 CSV로 저장되었습니다.")

if __name__ == "__main__":
    main()