#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 kNN Purity + t-SNE 벤치마크 (Ablation Study) — 검수 후 버전
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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

warnings.filterwarnings('ignore')
# 최적의 디바이스 자동 선택
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# =============================================================================
# 1. 모델 로드 (KURE-v1)
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = SentenceTransformer("nlpai-lab/KURE-v1", device=DEVICE)

    
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
# 벡터 임베딩 기반 유사도 행렬 생성
# =============================================================================
def embedding_cosine_matrix(token_lists):
    """
    KURE-v1 모델을 사용하되, 길이가 긴 문서는 쪼개서 임베딩 후 평균 풀링(Mean Pooling) 수행
    """
    sentences = [" ".join(tl) for tl in token_lists]
    if not any(s.strip() for s in sentences):
        return np.zeros((len(sentences), len(sentences)))

    # 모델의 최대 수용 길이 확인 (보통 512)
    max_length = EMB_MODEL.max_seq_length 
    all_embeddings = []

    for text in sentences:
        # 1. 텍스트를 토큰화 (길이 확인용)
        tokens = EMB_MODEL.tokenizer.tokenize(text)
        
        if len(tokens) <= max_length:
            # 길이가 짧으면 바로 임베딩
            emb = EMB_MODEL.encode(text, convert_to_tensor=True, show_progress_bar=False)
        else:
            # 2. 길면 max_length 단위로 쪼개기 (Chunking)
            chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
            chunk_texts = [EMB_MODEL.tokenizer.convert_tokens_to_string(c) for c in chunks]
            
            # 3. 각 덩어리별 임베딩 후 평균(Mean) 계산
            chunk_embs = EMB_MODEL.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=False)
            emb = torch.mean(chunk_embs, dim=0) # 이것이 '평균 풀링'
        
        all_embeddings.append(emb.cpu().numpy())

    # 리스트를 넘파이 행렬로 변환
    embeddings_stack = np.stack(all_embeddings)
    
    # 4. 코사인 유사도 계산
    sim_matrix = cosine_similarity(embeddings_stack)
    return sim_matrix

# [수정] 긴 문장 대응형 임베딩 추출 함수
def get_embeddings_chunked(sentences):
    max_length = EMB_MODEL.max_seq_length  # 보통 512
    all_embs = []
    
    for text in tqdm(sentences, desc="    Embedding Docs", leave=False):
        # 토큰화
        tokens = EMB_MODEL.tokenizer.tokenize(text)
        
        if len(tokens) <= max_length:
            # 짧은 문장은 바로 처리
            emb = EMB_MODEL.encode(text, convert_to_tensor=True, show_progress_bar=False)
        else:
            # 512 넘으면 쪼개서 평균 풀링
            chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
            chunk_texts = [EMB_MODEL.tokenizer.convert_tokens_to_string(c) for c in chunks]
            chunk_embs = EMB_MODEL.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=False)
            emb = torch.mean(chunk_embs, dim=0)
            
        all_embs.append(emb.cpu().numpy())
    return np.stack(all_embs)

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

    # ── 검수 후 버전: 검수 후 목록/final.txt 파일에서 로드 ──────────────
    reviewed = os.path.join(base_dir, "검수 후 목록")
    ngram_path = os.path.join(reviewed, f"stopwords_ngram_{domain}_final.txt")
    ngrams_all = _load_ngram_list(ngram_path)

    # 도메인별 검수 후 불용어 세트 로드 (A~E)
    CANDIDATES = {
        "A": f"stopwords_단일어_{domain}_A_zipf_final.txt",
        "B": f"stopwords_단일어_{domain}_B_entropy_final.txt",
        "C": f"stopwords_단일어_{domain}_C_mpvp_final.txt",
        "D": f"stopwords_단일어_{domain}_D_union_final.txt",
        "E": f"stopwords_단일어_{domain}_E_intersection_final.txt",
    }
    word_sets = {}
    for char, fname in CANDIDATES.items():
        path = os.path.join(reviewed, fname)
        word_sets[char] = _load_txt(path)

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

# [복구] t-SNE 시각화 함수 (KURE 임베딩 기반)
def plot_tsne(embeddings, labels, config_name, domain, out_dir):
    try:
        # t-SNE 연산 (임베딩 벡터 직접 입력)
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
        X2 = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(9, 7))
        for cat, color in DOMAIN_COLORS.items():
            idx = [i for i, l in enumerate(labels) if l == cat]
            if idx:
                ax.scatter(X2[idx, 0], X2[idx, 1], c=color, label=cat, alpha=0.6, s=30, edgecolors='none')
        
        ax.set_title(f"KURE-v1 t-SNE: {domain}\n({config_name})", fontsize=13)
        ax.legend(title="법역", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(out_dir, f"tsne_{domain}_{config_name.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"      ⚠️ t-SNE 시각화 중 에러 발생: {e}")

# =============================================================================
# [복구]  시각화 핵심 로직
# =============================================================================
# [수정] 벤치마크 실행 함수 (P@5~20 및 t-SNE 통합)
def run_enhanced_benchmark(dataset_name, df, configs):
    sample = balanced_sample(df, "법역", N_SAMPLE, RANDOM_SEED)
    labels = sample["법역"].values
    
    results = []
    # 시각화할 주요 Ablation 포인트 정의
    visualize_targets = ["00. Baseline (No SW)", "02. KIL Only", "24-E. KIL + 법령E + ngram", "24-E. KIL + 판례E + ngram"]

    for name in sorted(configs.keys()):
        print(f"  [Processing] {name}...")
        config = configs[name]
        token_lists = [preprocess(row["text"], config) for _, row in sample.iterrows()]
        sentences = [" ".join(tl) for tl in token_lists]

        # 1. Jaccard (어휘 기반)
        sim_jaccard = jaccard_matrix(token_lists)
        
        # 2. KURE (의미 기반 + 평균 풀링)
        # 텍스트가 다 날아간 경우 대비
        if not any(s.strip() for s in sentences):
            embeddings = np.zeros((len(sentences), 768)) # 모델 히든 사이즈 기본 768
        else:
            embeddings = get_embeddings_chunked(sentences)
        
        sim_embed = cosine_similarity(embeddings)

        # 3. 결과 기록
        res = {"설정": name}
        for k in [5, 10, 20]:
            res[f"Jac_P@{k}"] = round(knn_purity(sim_jaccard, labels, k), 4)
            res[f"KURE_P@{k}"] = round(knn_purity(sim_embed, labels, k), 4)
        results.append(res)

        # 4. 특정 설정일 때 t-SNE 그리기
        if any(target in name for target in visualize_targets):
            print(f"    -> Generating t-SNE plot...")
            plot_tsne(embeddings, labels, name, dataset_name, TSNE_DIR)
        
    return pd.DataFrame(results)


# =============================================================================
# 메인 실행부 (Main)
# =============================================================================
if __name__ == "__main__":
    _setup_font()
    pd.options.display.float_format = '{:.4f}'.format
    
    # 디렉토리 생성 확인
    if not os.path.exists(TSNE_DIR): os.makedirs(TSNE_DIR)

    print("\n" + "=" * 60)
    print("  kNN Purity & t-SNE Ablation Study")
    print("  (Chunking + Mean Pooling Enabled)")
    print("=" * 60)

    # 1. 법령 데이터
    print("\n[1/2] 법령 데이터 처리 중...")
    law_df = load_law_data()
    law_configs = build_configs_for_domain(BASE_DIR, "법령")
    law_results = run_enhanced_benchmark("법령", law_df, law_configs)

    # 결과 테이블 출력
    print("\n" + "=" * 95)
    print(f"  {'법령 설정':<35} | {'Jac@10':<8} {'KURE@10':<8} | {'Jac@20':<8} {'KURE@20':<8}")
    print("-" * 95)
    for _, row in law_results.iterrows():
        print(f"  {row['설정']:<35} | {row['Jac_P@10']:.4f}  {row['KURE_P@10']:.4f} | {row['Jac_P@20']:.4f}  {row['KURE_P@20']:.4f}")
    print("=" * 95)
    law_results.to_csv(os.path.join(BASE_DIR, "07.benchmark_knn_법령.csv"), index=False, encoding="utf-8-sig")

    # 2. 판례 데이터
    print("\n\n[2/2] 판례 데이터 처리 중...")
    case_df = load_case_data()
    case_configs = build_configs_for_domain(BASE_DIR, "판례")
    case_results = run_enhanced_benchmark("판례", case_df, case_configs)

    # 결과 테이블 출력
    print("\n" + "=" * 95)
    print(f"  {'판례 설정':<35} | {'Jac@10':<8} {'KURE@10':<8} | {'Jac@20':<8} {'KURE@20':<8}")
    print("-" * 95)
    for _, row in case_results.iterrows():
        print(f"  {row['설정']:<35} | {row['Jac_P@10']:.4f}  {row['KURE_P@10']:.4f} | {row['Jac_P@20']:.4f}  {row['KURE_P@20']:.4f}")
    print("=" * 95)
    case_results.to_csv(os.path.join(BASE_DIR, "07.benchmark_knn_판례.csv"), index=False, encoding="utf-8-sig")

    print(f"\n✅ 완료! 결과 CSV 및 t-SNE 이미지가 '{TSNE_DIR}'에 저장되었습니다.")