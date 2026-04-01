#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
n-gram 상투 구문 후보 추출 스크립트
=====================================
법령/판례 코퍼스에서 bigram·trigram(최대 3-gram)을 추출하고
문서 빈도(DF) 기준으로 정렬하여 수작업 검토용 CSV를 생성합니다.
"""

import os
import re
import json
import math
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# =============================================================================
# 설정
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "stopword_ngram_후보")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAW_CSV_DIR = os.path.join(BASE_DIR, "data", "법령 csv 저장(26.03.16 기준) 중복제거")
CASE_DIR    = os.path.join(BASE_DIR, "data", "대법원_판례")
LAW_LIST    = os.path.join(BASE_DIR, "00.전체_법률_목록(2026.03.16).csv")

# 평가 대상 법역 (제4편 행정일반, 제8편 민사법, 제9편 형사법)
TARGET_DOMAIN_CODES = [(4000000, 4999999), (8000000, 8999999), (9000000, 9999999)]

# 판례 대상 사건종류
TARGET_CASE_TYPES   = {"민사", "형사", "일반행정"}

# n-gram 설정
MIN_N, MAX_N   = 2, 3       # bigram ~ trigram
MIN_DF_LAW     = 10         # 법령: 최소 10개 법률에서 출현
MIN_DF_CASE    = 30         # 판례: 최소 30개 판례에서 출현
TOP_N          = 300        # 출력할 후보 수


# =============================================================================
# 엔트로피 계산 함수 (추가됨)
# =============================================================================
def calculate_entropy(doc_tf_dict: dict) -> float:
    """
    특정 n-gram이 출현한 문서별 빈도 딕셔너리를 받아 정보 엔트로피를 계산합니다.
    """
    total_tf = sum(doc_tf_dict.values())
    if total_tf <= 0:
        return 0.0
        
    entropy = 0.0
    for tf in doc_tf_dict.values():
        p = tf / total_tf  # 전체 출현 횟수 중 해당 문서에서 출현한 비율
        entropy -= p * math.log2(p)
        
    return entropy

# =============================================================================
# 텍스트 전처리
# =============================================================================

def clean_text(text: str) -> str:
    """HTML 태그, 조문번호 헤더, 특수문자 제거 후 정규화"""
    text = re.sub(r'<[^>]+>', ' ', text)               # HTML 태그
    text = re.sub(r'제\d+조(?:의\d+)?(?:\([^)]*\))?', '', text)  # 조문번호
    text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]', '', text)      # 원문자 항번호
    text = re.sub(r'[\[\]「」『』《》<>【】]', ' ', text) # 괄호류
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def eojeol_split(text: str) -> list[str]:
    """어절 단위 분리 (2글자 이상 한글 포함 어절만)"""
    tokens = re.split(r'\s+', text)
    return [t for t in tokens if len(t) >= 2 and re.search(r'[가-힣]', t)]


def extract_ngrams(tokens: list[str], min_n: int, max_n: int) -> list[tuple]:
    """토큰 리스트에서 n-gram 튜플 목록 반환"""
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
    return ngrams


# =============================================================================
# 필터: 내용어 포함 n-gram 제거
# =============================================================================

# 내용어 패턴: 고유 법령명·기관명·수치 등이 포함된 n-gram은 상투어가 아님
CONTENT_PATTERNS = [
    re.compile(r'[A-Za-z]'),          # 영문자 포함
    re.compile(r'\d'),                # 숫자 포함
    re.compile(r'법$'),               # "○○법"으로 끝나는 토큰 (특정 법령명)
    re.compile(r'^[가-힣]{1}$'),      # 1글자 한글 단독 토큰
]

def has_content_word(ngram: tuple) -> bool:
    """내용어가 포함된 n-gram이면 True (후보에서 제외)"""
    for token in ngram:
        for pat in CONTENT_PATTERNS:
            if pat.search(token):
                return True
    return False


# =============================================================================
# 법령 n-gram 추출
# =============================================================================

def extract_law_ngrams() -> pd.DataFrame:
    print("\n[법령 n-gram 추출]")

    # 대상 법령명 목록
    법률목록 = pd.read_csv(LAW_LIST)
    대상법령 = set()
    for lo, hi in TARGET_DOMAIN_CODES:
        sub = 법률목록[법률목록['법령분야코드'].between(lo, hi)]['법령명']
        대상법령.update(sub.tolist())
    print(f"  대상 법률: {len(대상법령)}개")

    # (수정됨) set 대신 Counter를 저장하도록 변경
    ngram_docs = defaultdict(Counter)

    files = [f for f in os.listdir(LAW_CSV_DIR) if f.endswith('.csv')]
    target_files = [f for f in files if os.path.splitext(f)[0] in 대상법령]
    print(f"  CSV 파일: {len(target_files)}개")

    for fname in tqdm(target_files, desc="  법령 처리"):
        law_name = os.path.splitext(fname)[0]
        path = os.path.join(LAW_CSV_DIR, fname)
        try:
            df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
        except Exception:
            continue

        for text in df['조문내용'].dropna():
            cleaned = clean_text(str(text))
            tokens  = eojeol_split(cleaned)
            for gram in extract_ngrams(tokens, MIN_N, MAX_N):
                if not has_content_word(gram):
                    # 출현할 때마다 해당 법령의 빈도 증가
                    ngram_docs[gram][law_name] += 1

    # DataFrame 생성 (TF, Entropy 컬럼 포함)
    rows = []
    for gram, doc_counts in ngram_docs.items():
        df_count = len(doc_counts)
        if df_count >= MIN_DF_LAW:
            tf_count = sum(doc_counts.values())
            entropy = calculate_entropy(doc_counts)
            
            rows.append({
                "ngram":   " ".join(gram),
                "n":       len(gram),
                "tf":      tf_count,
                "df":      df_count,
                "entropy": entropy
            })

    # DF, Entropy 복합 기준으로 정렬
    df_out = (pd.DataFrame(rows)
              .sort_values(by=["df", "entropy"], ascending=[False, False])
              .reset_index(drop=True)
              .head(TOP_N))

    print(f"  DF≥{MIN_DF_LAW} 후보: {len(df_out)}개")
    return df_out


# =============================================================================
# 판례 n-gram 추출
# =============================================================================

def extract_case_ngrams() -> pd.DataFrame:
    print("\n[판례 n-gram 추출]")

    files = [f for f in os.listdir(CASE_DIR) if f.endswith('.json')]
    print(f"  전체 판례: {len(files)}개")

    ngram_docs = defaultdict(Counter)
    used = 0

    for fname in tqdm(files, desc="  판례 처리"):
        path = os.path.join(CASE_DIR, fname)
        try:
            with open(path, encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue

        if obj.get('사건종류명') not in TARGET_CASE_TYPES:
            continue

        # 판결요지 우선, 없으면 판시사항
        text = obj.get('판결요지') or obj.get('판시사항') or ''
        if not text:
            continue

        used += 1
        case_id = obj.get('사건번호', fname)
        cleaned = clean_text(str(text))
        tokens  = eojeol_split(cleaned)

        for gram in extract_ngrams(tokens, MIN_N, MAX_N):
            if not has_content_word(gram):
                ngram_docs[gram][case_id] += 1

    print(f"  사용 판례: {used}개")

    rows = []
    for gram, doc_counts in ngram_docs.items():
        df_count = len(doc_counts)
        if df_count >= MIN_DF_CASE:
            tf_count = sum(doc_counts.values())
            entropy = calculate_entropy(doc_counts)
            
            rows.append({
                "ngram":   " ".join(gram),
                "n":       len(gram),
                "tf":      tf_count,
                "df":      df_count,
                "entropy": entropy
            })

    df_out = (pd.DataFrame(rows)
              .sort_values(by=["df", "entropy"], ascending=[False, False])
              .reset_index(drop=True)
              .head(TOP_N))

    print(f"  DF≥{MIN_DF_CASE} 후보: {len(df_out)}개")
    return df_out


# =============================================================================
# 메인
# =============================================================================

def main():
    print("=" * 60)
    print(" n-gram 상투 구문 후보 추출 (Entropy 적용)")
    print("=" * 60)
    print(f"  bigram ~ {MAX_N}-gram  |  법령 min DF={MIN_DF_LAW}  |  판례 min DF={MIN_DF_CASE}")

    # 법령
    df_law = extract_law_ngrams()
    out_law = os.path.join(OUTPUT_DIR, "ngram_candidates_법령.csv")
    df_law.to_csv(out_law, index=False, encoding='utf-8-sig')
    print(f"\n  저장: {out_law}")
    print("\n[법령 상위 30개]")
    print(df_law.head(30).to_string(index=False))

    # 판례
    df_case = extract_case_ngrams()
    out_case = os.path.join(OUTPUT_DIR, "ngram_candidates_판례.csv")
    df_case.to_csv(out_case, index=False, encoding='utf-8-sig')
    print(f"\n  저장: {out_case}")
    print("\n[판례 상위 30개]")
    print(df_case.head(30).to_string(index=False))

    print("\n" + "=" * 60)
    print(" 다음 단계")
    print("=" * 60)
    print(f"""
  1. CSV 파일을 열어 상투 구문 후보 검토
       {out_law}
       {out_case}

  2. 내용어가 포함된 구문, 의미 있는 구문 제거 (수작업)
     (팁: 'entropy'가 높을수록 상투어일 확률이 높습니다!)

  3. 최종 목록을 한 줄에 구문 하나씩 저장:
       stopwords_ngram_법령.txt
       stopwords_ngram_판례.txt
""")


if __name__ == "__main__":
    main()