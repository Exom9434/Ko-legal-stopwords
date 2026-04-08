#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 kNN Purity + t-SNE 벤치마크 (Ablation Study)
===================================================
1. 법령과 판례의 평가 지표 통일 (법역 Purity 중심 1:1 비교)
2. 도메인별 독립 설정 (크로스 없음)
3. BM25와 동일한 Ablation 경우의 수 (ngram only, General+ngram 등)
4. 최종 결과를 하나의 통합된 비교 표로 출력
"""
from konlpy.tag import Mecab
import os, re, json, random, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# =============================================================================
# 설정 (기존과 동일하되 직관적으로 정리)
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LAW_CSV_DIR = os.path.join(BASE_DIR, "data", "법령 csv 저장(26.03.16 기준) 중복제거")
CASE_DIR    = os.path.join(BASE_DIR, "data", "대법원_판례")
LAW_LIST    = os.path.join(BASE_DIR, "00.전체_법률_목록(2026.03.16).csv")
TSNE_DIR    = os.path.join(BASE_DIR, "tsne_plots")
os.makedirs(TSNE_DIR, exist_ok=True)

DOMAIN_CODES = {"행정일반": (4000000, 4999999), "민사법": (8000000, 8999999), "형사법": (9000000, 9999999)}
CASE_TYPE_MAP = {"일반행정": "행정일반", "민사": "민사법", "형사": "형사법"}

K_VALUES    = [5, 10, 20]
N_SAMPLE    = 300
RANDOM_SEED = 42

# t-SNE 시각화 대상: Baseline → KIL → KIL+E → KIL+E+ngram 흐름으로 Ablation 효과 극대화
TSNE_CONFIGS = ["00. Baseline", "02. KIL", "07. KIL + E", "11. KIL + E + ngram"]

def _setup_font():
    for font in ["NanumGothic", "AppleGothic", "Malgun Gothic", "NanumBarunGothic"]:
        try:
            fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
            plt.rcParams["font.family"] = font
            return
        except Exception:
            pass

def _load_txt(path): return {line.strip() for line in open(path, encoding="utf-8") if line.strip()} if os.path.exists(path) else set()
def _load_csv_words(path): return set(pd.read_csv(path)["word"].astype(str)) if os.path.exists(path) else set()
def _load_ngram_list(path): return [line.strip() for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []

# =============================================================================
# t-SNE 시각화 함수 (이전과 동일, 확인용)
# =============================================================================
def plot_tsne(token_lists, labels, config_name, domain, out_dir):
    docs = [" ".join(tl) for tl in token_lists]
    if not any(docs): return

    try:
        # TF-IDF 벡터화
        X = TfidfVectorizer(max_features=5000, min_df=2).fit_transform(docs).toarray()
        
        # t-SNE 임베딩
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
        X2 = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 7))
        for cat, color in DOMAIN_COLORS.items():
            idx = [i for i, l in enumerate(labels) if l == cat]
            if idx:
                ax.scatter(X2[idx, 0], X2[idx, 1], c=color, label=cat, alpha=0.7, s=25, edgecolors='none')
        
        ax.set_title(f"t-SNE Clustering: {domain} ({config_name})")
        ax.legend(title="법역", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(out_dir, f"tsne_{domain}_{config_name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"      ⚠️ t-SNE 에러 ({config_name}): {e}")

# =============================================================================
# [개선] 도메인별 독립 Config 빌더 (크로스 없음, BM25와 동일한 Ablation 구조)
# =============================================================================
def build_configs_for_domain(base_dir: str, domain: str) -> dict:
    """
    domain: '법령' or '판례'
    BM25 전수조사 버전과 동일한 51개 내외의 실험군 생성
    """
    kil = _load_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = _load_txt(os.path.join(base_dir, "stopwords.txt"))

    stat_dir   = os.path.join(base_dir, f"stopword_{domain}")
    ngram_path = os.path.join(base_dir, f"stopwords_ngram_{domain}.txt")
    ngrams_all = _load_ngram_list(ngram_path)

    # 도메인별 불용어 세트 로드 (A~E)
    CANDIDATES = {
        "A": "candidates_A_zipf.csv",
        "B": "candidates_B_entropy.csv",
        "C": "candidates_C_mpvp.csv",
        "D": "candidates_D_union.csv",
        "E": "candidates_E_intersection.csv",
    }
    word_sets = {}
    for char, fname in CANDIDATES.items():
        path = os.path.join(stat_dir, fname)
        word_sets[char] = _load_csv_words(path)

    cfgs = {}

    # ==========================================
    # [그룹 0-1] 기초 및 N-gram 단독 효과 (00-07)
    # ==========================================
    cfgs["00. Baseline (No SW)"] = {"words": set(), "ngrams": []}
    cfgs["01. General (SW)"]     = {"words": gen,   "ngrams": []}
    cfgs["02. KIL Only"]         = {"words": kil,   "ngrams": []}
    cfgs["03. Baseline + ngram-050"] = {"words": set(), "ngrams": ngrams_all[:50]}
    cfgs["04. Baseline + ngram-100"] = {"words": set(), "ngrams": ngrams_all[:100]}
    cfgs["05. Baseline + ngram-150"] = {"words": set(), "ngrams": ngrams_all[:150]}
    cfgs["06. Baseline + ngram-200"] = {"words": set(), "ngrams": ngrams_all[:200]}
    cfgs["07. Baseline + ngram-all"] = {"words": set(), "ngrams": ngrams_all}

    # ==========================================
    # [그룹 2-3] 도메인 불용어 단독 및 KIL 결합 (08-17)
    # ==========================================
    for char, sw_set in word_sets.items():
        cfgs[f"08-{char}. Baseline + {domain}{char}"] = {"words": sw_set, "ngrams": []}
    for char, sw_set in word_sets.items():
        cfgs[f"13-{char}. KIL + {domain}{char}"]      = {"words": kil | sw_set, "ngrams": []}

    # ==========================================
    # [그룹 4-5] KIL/Baseline + N-gram 풀조합 (18-28)
    # ==========================================
    cfgs[f"18. KIL + {domain}ngram"] = {"words": kil, "ngrams": ngrams_all}
    for char, sw_set in word_sets.items():
        cfgs[f"19-{char}. Baseline + {domain}{char} + ngram"] = {"words": sw_set, "ngrams": ngrams_all}
    for char, sw_set in word_sets.items():
        cfgs[f"24-{char}. KIL + {domain}{char} + ngram"]      = {"words": kil | sw_set, "ngrams": ngrams_all}

    # ==========================================
    # [그룹 6-7] General SW 기반 대조군 (29-38)
    # ==========================================
    for char, sw_set in word_sets.items():
        cfgs[f"29-{char}. General + {domain}{char}"] = {"words": gen | sw_set, "ngrams": []}
    for char, sw_set in word_sets.items():
        cfgs[f"34-{char}. General + {domain}{char} + ngram"] = {"words": gen | sw_set, "ngrams": ngrams_all}

    # ==========================================
    # [그룹 8-10] 크기별 정밀 분석 (39-50)
    # ==========================================
    sizes = [50, 100, 150, 200]
    # KIL 기반
    for i, n in enumerate(sizes):
        cfgs[f"{39+i:02d}. KIL + ngram-{n:03d}"] = {"words": kil, "ngrams": ngrams_all[:n]}
    # Baseline + E 기반
    for i, n in enumerate(sizes):
        cfgs[f"{43+i:02d}. Baseline + {domain}E + ngram-{n:03d}"] = {"words": word_sets.get("E", set()), "ngrams": ngrams_all[:n]}
    # General 기반
    for i, n in enumerate(sizes):
        cfgs[f"{47+i:02d}. General + ngram-{n:03d}"] = {"words": gen, "ngrams": ngrams_all[:n]}

    return cfgs

# =============================================================================
# 텍스트 전처리 및 데이터 로더 (기존 로직 유지하되 간결화)
# =============================================================================
mecab = Mecab(dicpath='/opt/homebrew/lib/mecab/dic/mecab-ko-dic')

def preprocess(text: str, config: dict) -> list:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for phrase in config["ngrams"]:
        text = text.replace(phrase, " ")
    tokens = [word for word, pos in mecab.pos(text) if pos.startswith("N")]
    return [t for t in tokens if len(t) >= 2 and t not in config["words"]]

def load_law_data() -> pd.DataFrame:
    법률목록 = pd.read_csv(LAW_LIST)
    records = []
    TEXT_COLS = ["조문내용", "항", "호", "목"]
    for 법역명, (lo, hi) in DOMAIN_CODES.items():
        법령들 = 법률목록[법률목록["법령분야코드"].between(lo, hi)]["법령명"].tolist()
        for 법령명 in 법령들:
            path = os.path.join(LAW_CSV_DIR, f"{법령명}.csv")
            if not os.path.exists(path): continue
            try: df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
            except: continue
            if "조문번호" not in df.columns: continue
            use_cols = [c for c in TEXT_COLS if c in df.columns]
            for art_num, group in df.groupby("조문번호", sort=False):
                parts = [str(val).strip() for col in use_cols for val in group[col] if pd.notna(val) and len(str(val).strip()) >= 2]
                text = " ".join(parts)
                if len(text) >= 20: records.append({"text": text, "법령명": 법령명, "법역": 법역명})
    return pd.DataFrame(records)

def load_case_data() -> pd.DataFrame:
    records = []
    files = [f for f in os.listdir(CASE_DIR) if f.endswith(".json")]
    for fname in tqdm(files, desc="  판례 로드", leave=False):
        try: obj = json.load(open(os.path.join(CASE_DIR, fname), encoding="utf-8"))
        except: continue
        종류명 = obj.get("사건종류명", "")
        if 종류명 not in CASE_TYPE_MAP: continue
        text = obj.get("판결요지") or obj.get("판시사항") or ""
        text = re.sub(r"<br/?>", " ", text).strip()
        if len(text) >= 30:
            records.append({"text": text, "사건번호": obj.get("사건번호", fname), "법역": CASE_TYPE_MAP[종류명]})
    return pd.DataFrame(records)

def balanced_sample(df, label_col, n, seed):
    rng = random.Random(seed)
    sampled = []
    for label in df[label_col].unique():
        group = df[df[label_col] == label]
        sampled.extend(rng.sample(list(group.index), min(n, len(group))))
    return df.loc[sampled].reset_index(drop=True)

# =============================================================================
# Jaccard 유사도 + kNN Purity 측정
# =============================================================================
def jaccard_matrix(token_lists):
    n = len(token_lists)
    sets = [frozenset(tl) for tl in token_lists]
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            sim[i, j] = sim[j, i] = len(sets[i] & sets[j]) / union if union > 0 else 0.0
    return sim

def knn_purity(sim_matrix, labels, k):
    n = len(labels)
    purities = []
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -1.0
        top_k = np.argsort(row)[::-1][:k]
        purities.append(sum(1 for j in top_k if labels[j] == labels[i]) / k)
    return float(np.mean(purities))

# =============================================================================
# t-SNE 시각화
# =============================================================================
DOMAIN_COLORS = {"행정일반": "#E74C3C", "민사법": "#3498DB", "형사법": "#2ECC71"}

def plot_tsne(token_lists, labels, config_name, domain, out_dir):
    docs = [" ".join(tl) for tl in token_lists]
    if all(len(d.strip()) == 0 for d in docs): return

    try: X = TfidfVectorizer(max_features=5000, min_df=2).fit_transform(docs).toarray()
    except: return

    X2 = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, len(docs) // 4), max_iter=1000, verbose=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for 법역명, color in DOMAIN_COLORS.items():
        idx = [i for i, l in enumerate(labels) if l == 법역명]
        if idx: ax.scatter(X2[idx, 0], X2[idx, 1], c=color, label=법역명, alpha=0.6, s=15, linewidths=0)
    
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"t-SNE — {domain} — {config_name}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tsne_{domain}_{config_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# [복구] t-SNE 시각화 핵심 로직
# =============================================================================
def run_enhanced_benchmark(dataset_name, df, configs):
    """
    dataset_name: '법령' or '판례'
    configs: build_configs_for_domain()로 만든 도메인 전용 설정 (크로스 없음)
    """
    sample = balanced_sample(df, "법역", N_SAMPLE, RANDOM_SEED)
    labels = sample["법역"].values
    law_labels = sample["법령명"].values if "법령명" in sample.columns else None

    results = []
    for name in sorted(configs.keys()):
        config = configs[name]

        token_lists = [preprocess(row["text"], config) for _, row in sample.iterrows()]

        sim = jaccard_matrix(token_lists)
        res = {"설정": name, "P@5":  round(knn_purity(sim, labels, 5),  4),
                              "P@10": round(knn_purity(sim, labels, 10), 4),
                              "P@20": round(knn_purity(sim, labels, 20), 4)}
        if law_labels is not None:
            res["법령_P@10"] = round(knn_purity(sim, law_labels, 10), 4)
        results.append(res)

        # t-SNE: Baseline → KIL → KIL+E → KIL+E+ngram 흐름
        if any(name.startswith(t) for t in TSNE_CONFIGS):
            print(f"    [t-SNE] {dataset_name} - {name} 시각화 생성 중...")
            plot_tsne(token_lists, list(labels), name, dataset_name, TSNE_DIR)

    return pd.DataFrame(results)


# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    _setup_font()
    pd.options.display.float_format = '{:.4f}'.format
    print("=" * 60)
    print("  kNN Purity Ablation Study")
    print("=" * 60)

    # ── 법령 벤치마크 ──────────────────────────────────────────
    print("\n[1/2] 법령 데이터 로드 중...")
    law_df = load_law_data()
    print(f"  → {len(law_df):,}개 조문 로드 완료")

    law_configs = build_configs_for_domain(BASE_DIR, "법령")
    print(f"  → 설정 {len(law_configs)}개: {list(law_configs.keys())}\n")

    print("  법령 kNN Purity 계산 중...")
    law_results = run_enhanced_benchmark("법령", law_df, law_configs)

    print("\n" + "=" * 65)
    print(f"  {'설정':<28} | P@5    P@10   P@20")
    print("=" * 65)
    for _, row in law_results.iterrows():
        print(f"  {row['설정']:<28} | {row['P@5']:.4f} {row['P@10']:.4f} {row['P@20']:.4f}")

    law_results.to_csv(os.path.join(BASE_DIR, "07.benchmark_knn_법령.csv"), index=False, encoding="utf-8-sig")

    # ── 판례 벤치마크 ──────────────────────────────────────────
    print("\n\n[2/2] 판례 데이터 로드 중...")
    case_df = load_case_data()
    print(f"  → {len(case_df):,}개 판례 로드 완료")

    case_configs = build_configs_for_domain(BASE_DIR, "판례")
    print(f"  → 설정 {len(case_configs)}개: {list(case_configs.keys())}\n")

    print("  판례 kNN Purity 계산 중...")
    case_results = run_enhanced_benchmark("판례", case_df, case_configs)

    print("\n" + "=" * 65)
    print(f"  {'설정':<28} | P@5    P@10   P@20")
    print("=" * 65)
    for _, row in case_results.iterrows():
        print(f"  {row['설정']:<28} | {row['P@5']:.4f} {row['P@10']:.4f} {row['P@20']:.4f}")

    case_results.to_csv(os.path.join(BASE_DIR, "07.benchmark_knn_판례.csv"), index=False, encoding="utf-8-sig")

    print("\n✅ 완료! CSV 저장됨.")