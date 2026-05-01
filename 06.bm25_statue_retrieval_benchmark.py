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
    19: ["docid-9908__dumpdate-20231128___jo-212"], 
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
    # 기초 불용어 로드
    kil = load_stopwords_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = load_stopwords_txt(os.path.join(base_dir, "stopwords.txt"))
    
    # 1. 법령 불용어 CSV 로드
    stat_dir_law = os.path.join(base_dir, "stopword_법령")
    law_a = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_A_zipf.csv"))
    law_b = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_B_entropy.csv"))
    law_c = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_C_mpvp.csv"))
    law_d = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_D_union.csv"))
    law_e = load_stopwords_csv(os.path.join(stat_dir_law, "candidates_E_intersection.csv"))
    
    # 2. 법령 N-gram 로드
    ngram_law = _load_ngram_list(os.path.join(base_dir, "stopwords_ngram_법령.txt"))

    cfgs = {}

    # ==========================================
    # [그룹 0-1] 기초 및 N-gram 단독 효과 (00-07)
    # ==========================================
    cfgs["00. Baseline (No SW)"] = {"words": set(), "ngrams": []}
    cfgs["01. General (SW)"]     = {"words": gen, "ngrams": []}
    cfgs["02. KIL Only"]         = {"words": kil, "ngrams": []}
    cfgs["03. Baseline + 법령ngram-050"] = {"words": set(), "ngrams": ngram_law[:50]}
    cfgs["04. Baseline + 법령ngram-100"] = {"words": set(), "ngrams": ngram_law[:100]}
    cfgs["05. Baseline + 법령ngram-150"] = {"words": set(), "ngrams": ngram_law[:150]}
    cfgs["06. Baseline + 법령ngram-200"] = {"words": set(), "ngrams": ngram_law[:200]}
    cfgs["07. Baseline + 법령ngram-all"] = {"words": set(), "ngrams": ngram_law}

    # ==========================================
    # [그룹 2-3] 도메인 불용어 단독 및 KIL 결합 (08-17)
    # ==========================================
    laws = {"A": law_a, "B": law_b, "C": law_c, "D": law_d, "E": law_e}
    for char, sw_set in laws.items():
        cfgs[f"08-{char}. Baseline + 법령{char}"] = {"words": sw_set, "ngrams": []}
    for char, sw_set in laws.items():
        cfgs[f"13-{char}. KIL + 법령{char}"]      = {"words": kil | sw_set, "ngrams": []}

    # ==========================================
    # [그룹 4-5] KIL/Baseline + N-gram 풀조합 (18-28)
    # ==========================================
    cfgs["18. KIL + 법령ngram"] = {"words": kil, "ngrams": ngram_law}
    for char, sw_set in laws.items():
        cfgs[f"19-{char}. Baseline + 법령{char} + 법령ngram"] = {"words": sw_set, "ngrams": ngram_law}
    for char, sw_set in laws.items():
        cfgs[f"24-{char}. KIL + 법령{char} + 법령ngram"]      = {"words": kil | sw_set, "ngrams": ngram_law}

    # ==========================================
    # [그룹 6-7] General SW 기반 대조군 (29-38)
    # ==========================================
    # KIL(학술용)이 아닌 일반적인 불용어를 썼을 때 도메인 불용어의 효율 측정
    for char, sw_set in laws.items():
        cfgs[f"29-{char}. General + 법령{char}"] = {"words": gen | sw_set, "ngrams": []}
    for char, sw_set in laws.items():
        cfgs[f"34-{char}. General + 법령{char} + 법령ngram"] = {"words": gen | sw_set, "ngrams": ngram_law}

    # ==========================================
    # [그룹 8] KIL + N-gram 크기별 정밀 분석 (39-42)
    # ==========================================
    cfgs["39. KIL + 법령ngram-050"] = {"words": kil, "ngrams": ngram_law[:50]}
    cfgs["40. KIL + 법령ngram-100"] = {"words": kil, "ngrams": ngram_law[:100]}
    cfgs["41. KIL + 법령ngram-150"] = {"words": kil, "ngrams": ngram_law[:150]}
    cfgs["42. KIL + 법령ngram-200"] = {"words": kil, "ngrams": ngram_law[:200]}

    # ==========================================
    # [그룹 9] Baseline + 법령E(최적후보) + N-gram 크기별 (43-46)
    # ==========================================
    # KIL 없이 법령E와 N-gram 조합만으로 어디까지 성능이 나오는지 확인
    cfgs["43. Baseline + 법령E + 법령ngram-050"] = {"words": law_e, "ngrams": ngram_law[:50]}
    cfgs["44. Baseline + 법령E + 법령ngram-100"] = {"words": law_e, "ngrams": ngram_law[:100]}
    cfgs["45. Baseline + 법령E + 법령ngram-150"] = {"words": law_e, "ngrams": ngram_law[:150]}
    cfgs["46. Baseline + 법령E + 법령ngram-200"] = {"words": law_e, "ngrams": ngram_law[:200]}

    # ==========================================
    # [그룹 10] General + N-gram 크기별 (47-50)
    # ==========================================
    cfgs["47. General + 법령ngram-050"] = {"words": gen, "ngrams": ngram_law[:50]}
    cfgs["48. General + 법령ngram-100"] = {"words": gen, "ngrams": ngram_law[:100]}
    cfgs["49. General + 법령ngram-150"] = {"words": gen, "ngrams": ngram_law[:150]}
    cfgs["50. General + 법령ngram-200"] = {"words": gen, "ngrams": ngram_law[:200]}

    return cfgs
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

def normalize_law_name(text: str) -> str:
    """법률명 비교용 정규화: 공백/중점 차이를 흡수한다."""
    text = str(text or "")
    text = text.replace("·", "").replace("ㆍ", "")
    return re.sub(r"\s+", "", text)

def build_law_article_index(corpus: list[dict]) -> dict:
    """현재 statutes_local.jsonl ID 형식에서 {(법률명, 조번호): [corpus_id]} 인덱스를 만든다."""
    index = {}
    for doc in corpus:
        m = re.match(r"docid-\d+__(.+)__jo-(\d+)$", doc["id"])
        if not m:
            continue
        law_name = normalize_law_name(m.group(1))
        article_no = int(m.group(2))
        index.setdefault((law_name, article_no), []).append(doc["id"])
    return index

def parse_gold_ref(gold_ref: str):
    """
    예: '식품위생법 제36조1항', '형법 제232-2조', '민사소송법 제202조의2'
    현재 코퍼스는 가지조문을 부모 jo에 합쳐 둔 경우가 많으므로 첫 번째 조 번호만 사용한다.
    """
    m = re.search(r"(.+?)\s*제\s*(\d+)(?:\s*(?:-|의)\s*\d+)?\s*조", str(gold_ref or ""))
    if not m:
        return None
    return normalize_law_name(m.group(1)), int(m.group(2))

def load_kbl_queries(kbl_dir, task, corpus):
    path = os.path.join(kbl_dir, "tasks", f"{task}.jsonl")
    if not os.path.exists(path): return []

    print("  [알림] gold_ref 기반 정답 ID 매핑을 시작합니다...")
    law_article_index = build_law_article_index(corpus)

    queries = []
    matched, unmatched = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question") or ""
            if not q: continue

            gold_key = parse_gold_ref(obj.get("gold_ref", ""))
            gold_ids = law_article_index.get(gold_key, []) if gold_key else []
            if gold_ids:
                matched += 1
            else:
                unmatched += 1

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
