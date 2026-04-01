#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 kNN Purity + t-SNE 벤치마크 (Ablation Study)
===================================================
1. 법령과 판례의 평가 지표 통일 (법역 Purity 중심 1:1 비교)
2. 교차 검증(Cross-domain)이 포함된 체계적인 설정(Configs)
3. 최종 결과를 하나의 통합된 비교 표로 출력
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

# t-SNE는 Base(NONE)와 최종(Ngram 포함) 형태만 비교하여 Ablation 효과 극대화
TSNE_CONFIGS = ["1_NONE", "5_KIL+법령E+Ngram", "5_KIL+판례E+Ngram"]

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
# [개선] A, B, C, D, E 모든 경우의 수를 포함한 Config 빌더
# =============================================================================
def build_full_ablation_configs(base_dir: str) -> dict:
    kil = _load_txt(os.path.join(base_dir, "stopwords_kil(2018).txt"))
    gen = _load_txt(os.path.join(base_dir, "stopwords.txt"))
    
    configs = {
        "0_NONE":    {"words": set(), "ngrams": []},
        "1_GENERAL": {"words": gen, "ngrams": []},
        "2_KIL":     {"words": kil, "ngrams": []},
    }

    # 도메인별 A~E 후보군 로드 루프
    for domain in ["법령", "판례"]:
        stat_dir = os.path.join(base_dir, f"stopword_{domain}")
        ngram_path = os.path.join(base_dir, f"stopwords_ngram_{domain}.txt")
        ngrams = _load_ngram_list(ngram_path)
        
        # A: Zipf, B: TF-IDF, C: Entropy, D: POS-Skew, E: Intersection
        targets = {
            "A": "candidates_A_zipf.csv",
            "B": "candidates_B_tfidf.csv", # 추가
            "C": "candidates_C_entropy.csv", # 추가
            "D": "candidates_D_skew.csv",    # 추가
            "E": "candidates_E_intersection.csv"
        }

        for char, filename in targets.items():
            path = os.path.join(stat_dir, filename)
            if os.path.exists(path):
                words = _load_csv_words(path)
                # 기본 KIL + 도메인 특화 단어 조합
                configs[f"3_{domain}_{char}"] = {"words": kil | words, "ngrams": []}
                
                # 최상위 조합 (단어 E + Ngram) 예시 - 가장 성능이 좋을 것으로 기대되는 조합
                if char == "E" and ngrams:
                    configs[f"4_{domain}_E+Ngram"] = {"words": kil | words, "ngrams": ngrams}

    return configs

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
    sample = balanced_sample(df, "법역", N_SAMPLE, RANDOM_SEED)
    labels = sample["법역"].values
    law_labels = sample["법령명"].values if "법령명" in sample.columns else None

    # 시각화 대상 설정 (Ablation의 극적인 변화를 보여줄 포인트)
    # 0_NONE(기본), 2_KIL(기준), 3_도메인_E(단어 최적), 4_도메인_E+Ngram(최종)
    target_viz = ["0_NONE", "2_KIL", f"3_{dataset_name}_E", f"4_{dataset_name}_E+Ngram"]

    results = []
    for name in sorted(configs.keys()):
        config = configs[name]
        
        # 해당 도메인 실험이 아니면 스킵 (법령 테스트 시 판례 설정 제외 등)
        if dataset_name == "법령" and "판례_" in name: continue
        if dataset_name == "판례" and "법령_" in name: continue

        token_lists = [preprocess(row["text"], config) for _, row in sample.iterrows()]
        
        # Jaccard 및 Purity 계산
        sim = jaccard_matrix(token_lists)
        res = {"설정": name}
        res["P@10"] = round(knn_purity(sim, labels, 10), 4)
        if law_labels is not None:
            res["Detail_P@10"] = round(knn_purity(sim, law_labels, 10), 4)
        results.append(res)

        # --- t-SNE 시각화 실행부 복구 ---
        if any(target in name for target in target_viz):
            print(f"    [t-SNE] {dataset_name} - {name} 시각화 생성 중...")
            plot_tsne(token_lists, list(labels), name, dataset_name, TSNE_DIR)

    return pd.DataFrame(results)