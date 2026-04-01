#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean Legal Stopword Improvement Tool
법률 불용어 개선 도구

Based on classification benchmark results, this script improves stopword lists
by adding domain knowledge and filtering discriminative legal terms.
"""

import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from typing import Set, List, Dict
import os

# =============================================================================
# 1. LEGAL DOMAIN KNOWLEDGE
# =============================================================================

# 절대 제거하면 안 되는 법률 핵심 용어
LEGAL_PROTECTED_TERMS = {
    # 당사자 (Parties)
    '피고', '원고', '피의자', '피해자', '증인', '검사', '변호사', '판사',
    '피청구인', '청구인', '신청인', '피신청인', '항소인', '피항소인',
    '상고인', '피상고인', '고소인', '피고소인', '고발인', '피고발인',
    
    # 기관 (Institutions)
    '법원', '대법원', '고등법원', '지방법원', '가정법원', '행정법원',
    '특허법원', '검찰', '검찰청', '경찰', '경찰서',
    
    # 법률 문서/절차 (Legal Documents/Processes)
    '판결', '결정', '명령', '소송', '재판', '심판', '항소', '상고',
    '고소', '고발', '기소', '공소', '구속', '체포', '압수', '수색',
    '증거', '증명', '입증', '주장', '인정', '기각', '인용', '파기',
    '환송', '선고', '집행', '집행유예', '가처분', '가압류',
    
    # 법령 (Legal Basis)
    '법', '법률', '법령', '규정', '조례', '조', '항', '호', '목',
    '헌법', '민법', '형법', '상법', '형사소송법', '민사소송법',
    '행정소송법', '특허법', '저작권법', '근로기준법',
    
    # 권리/의무 (Rights/Obligations)
    '권리', '의무', '책임', '손해', '배상', '보상', '변제', '지급',
    '청구', '청구권', '소유권', '점유권', '채권', '채무', '계약',
    
    # 판결 결과 (Legal Outcomes)
    '유죄', '무죄', '벌금', '징역', '금고', '구류', '과료', '몰수',
    '추징', '형', '처벌', '형량',
    
    # 사건 유형 (Case Types) - CRITICAL for classification!
    '민사', '형사', '행정', '가사', '특허', '조세', '선거',
    
    # 시간/절차 (Time/Procedure)
    '기간', '기한', '기일', '통지', '송달', '공시', '공고', '공시송달',
    
    # 법률 행위 (Legal Acts)
    '계약', '합의', '조정', '화해', '중재', '위임', '대리',
    
    # 증거 관련
    '증인', '증언', '진술', '자백', '고백', '서증', '물증', '감정',
    
    # 기타 중요 용어
    '위법', '불법', '합법', '적법', '위헌', '합헌', '위반', '준수'
}

# 제거해도 괜찮은 법률 상투어 (Legal boilerplate OK to remove)
LEGAL_BOILERPLATE_SAFE = {
    '것', '등', '및', '또는', '또', '다만', '단',
    '경우', '때', '바', '대로', '따라', '의하여', '관하여', '대하여',
    '있어서', '대한', '관한', '의한', '위한', '관련',
    '이상', '이하', '미만', '이내', '사항', '내용', '방법', '절차'
}

# 일반 한국어 불용어 (순수 문법 요소만)
GENERAL_GRAMMAR_STOPWORDS = {
    # 조사 (Particles)
    '이', '가', '께서', '을', '를', '은', '는', '의',
    '에', '에게', '에서', '로', '으로', '부터', '까지',
    '와', '과', '하고', '도', '만', '조차', '마저',
    
    # 대명사 (Pronouns)
    '그', '이', '저', '이것', '그것', '저것', '여기', '거기', '저기',
    '나', '너', '우리', '저희',
    
    # 일반 부사 (Non-legal adverbs)
    '매우', '너무', '정말', '아주', '조금', '좀', '거의', '거의',
    '항상', '늘', '가끔', '때때로', '아마'
}


# =============================================================================
# 2. FILTERING FUNCTIONS
# =============================================================================

def filter_with_domain_knowledge(
    candidate_stopwords: Set[str],
    protected_terms: Set[str] = LEGAL_PROTECTED_TERMS
) -> Set[str]:
    """
    Remove legally important terms from stopword candidates
    
    Args:
        candidate_stopwords: 불용어 후보 집합
        protected_terms: 보호할 법률 용어 집합
    
    Returns:
        필터링된 불용어 집합
    """
    filtered = set(candidate_stopwords) - set(protected_terms)
    
    removed_count = len(candidate_stopwords) - len(filtered)
    if removed_count > 0:
        removed = set(candidate_stopwords) & set(protected_terms)
        print(f"  ⚠️  {removed_count}개 법률 핵심 용어 제거됨: {list(removed)[:10]}")
    
    return filtered


def get_pos_filtered_stopwords(
    candidate_stopwords: Set[str],
    allowed_pos_prefixes: List[str] = ['J', 'E', 'XS']
) -> Set[str]:
    """
    POS 태그 기반 필터링 - 조사, 어미, 접사만 허용
    
    Args:
        candidate_stopwords: 불용어 후보
        allowed_pos_prefixes: 허용할 품사 태그 접두사
            J: 조사 (Josa), E: 어미 (Eomi), XS: 접사 (Suffix)
    
    Returns:
        문법 요소만 포함하는 불용어 집합
    """
    mecab = Mecab()
    safe_stopwords = []
    
    for word in candidate_stopwords:
        try:
            pos = mecab.pos(word)
            if pos:
                pos_tag = pos[0][1]
                # 허용된 품사만 불용어로 인정
                if any(pos_tag.startswith(prefix) for prefix in allowed_pos_prefixes):
                    safe_stopwords.append(word)
        except Exception as e:
            # POS 태깅 실패 시 제외
            pass
    
    print(f"  📊 POS 필터링: {len(candidate_stopwords)}개 → {len(safe_stopwords)}개")
    
    return set(safe_stopwords)


def create_hybrid_stopwords(
    corpus_specific: Set[str],
    general_stopwords: Set[str] = GENERAL_GRAMMAR_STOPWORDS,
    use_intersection: bool = True
) -> Set[str]:
    """
    코퍼스 특화 불용어와 일반 불용어를 결합
    
    Args:
        corpus_specific: 코퍼스에서 추출한 불용어
        general_stopwords: 일반 한국어 불용어
        use_intersection: True면 교집합, False면 합집합
    
    Returns:
        하이브리드 불용어 집합
    """
    if use_intersection:
        # 보수적 접근: 두 리스트 모두에 있는 것만
        hybrid = corpus_specific.intersection(general_stopwords)
        print(f"  🔗 교집합 전략: {len(hybrid)}개")
    else:
        # 적극적 접근: 하나라도 있으면 포함
        hybrid = corpus_specific.union(general_stopwords)
        # 단, 법률 핵심 용어는 제외
        hybrid = hybrid - LEGAL_PROTECTED_TERMS
        print(f"  🔗 합집합 전략: {len(hybrid)}개")
    
    return hybrid


# =============================================================================
# 3. CLASS-CONDITIONAL ANALYSIS
# =============================================================================

def calculate_class_discriminative_score(
    word: str,
    corpus: List[str],
    labels: List[str]
) -> float:
    """
    단어의 클래스 판별력 계산
    
    높은 점수 = 클래스 간 빈도 차이 큼 = 판별력 높음 = 불용어 아님
    낮은 점수 = 클래스 간 빈도 차이 작음 = 판별력 낮음 = 불용어 후보
    
    Returns:
        Variance of word frequency across classes
    """
    from collections import defaultdict
    
    class_freqs = defaultdict(list)
    
    for doc, label in zip(corpus, labels):
        word_count = doc.split().count(word)
        doc_len = len(doc.split())
        freq = word_count / doc_len if doc_len > 0 else 0
        class_freqs[label].append(freq)
    
    # 클래스별 평균 빈도
    class_means = {cls: np.mean(freqs) for cls, freqs in class_freqs.items()}
    
    # 클래스 간 분산 (높을수록 판별력 높음)
    variance = np.var(list(class_means.values()))
    
    return variance


def get_class_aware_stopwords(
    candidates: Set[str],
    corpus: List[str],
    labels: List[str],
    variance_threshold: float = 0.0001
) -> Set[str]:
    """
    클래스 판별력 기반 불용어 선택
    
    Args:
        candidates: 불용어 후보
        corpus: 문서 리스트
        labels: 클래스 레이블 리스트
        variance_threshold: 이 값보다 낮으면 불용어로 간주
    
    Returns:
        클래스 비판별적 단어들 (진짜 불용어)
    """
    print(f"\n  🔬 클래스 판별력 분석 중... ({len(candidates)}개 후보)")
    
    true_stopwords = []
    discriminative_words = []
    
    for word in candidates:
        score = calculate_class_discriminative_score(word, corpus, labels)
        
        if score < variance_threshold:
            true_stopwords.append(word)
        else:
            discriminative_words.append((word, score))
    
    print(f"  ✅ 비판별적 (불용어): {len(true_stopwords)}개")
    print(f"  ⚠️  판별적 (보존): {len(discriminative_words)}개")
    
    if discriminative_words:
        # 가장 판별력 높은 단어들 출력
        top_discriminative = sorted(discriminative_words, key=lambda x: x[1], reverse=True)[:10]
        print(f"  📌 판별력 높은 단어 (불용어 아님): {[w for w, s in top_discriminative]}")
    
    return set(true_stopwords)


# =============================================================================
# 4. MAIN IMPROVEMENT PIPELINE
# =============================================================================

def improve_stopwords(
    exp_a: Set[str],  # Zipf 결과
    exp_b: Set[str],  # Entropy 결과  
    exp_c: Set[str],  # MP-VP 결과
    corpus: List[str] = None,  # 선택: 클래스 분석용
    labels: List[str] = None,  # 선택: 클래스 분석용
    output_dir: str = "improved_stopwords"
) -> Dict[str, Set[str]]:
    """
    Main improvement pipeline
    
    Returns:
        Dictionary of improved stopword variants
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🔧 Korean Legal Stopwords Improvement Pipeline")
    print("="*70)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Variant 1: Original Intersection (Exp E) - Baseline
    # -------------------------------------------------------------------------
    print("\n[1] Original Exp E (Intersection)")
    exp_e_original = exp_a.intersection(exp_b).intersection(exp_c)
    print(f"  크기: {len(exp_e_original)}개")
    results['V1_ExpE_Original'] = exp_e_original
    
    # -------------------------------------------------------------------------
    # Variant 2: Domain-Filtered Exp E
    # -------------------------------------------------------------------------
    print("\n[2] Domain-Filtered Exp E (법률 핵심 용어 제거)")
    exp_e_filtered = filter_with_domain_knowledge(exp_e_original)
    print(f"  크기: {len(exp_e_filtered)}개")
    results['V2_ExpE_DomainFiltered'] = exp_e_filtered
    
    # -------------------------------------------------------------------------
    # Variant 3: POS-Filtered Exp E (Grammar Only)
    # -------------------------------------------------------------------------
    print("\n[3] POS-Filtered Exp E (문법 요소만)")
    exp_e_pos_filtered = get_pos_filtered_stopwords(exp_e_original)
    print(f"  크기: {len(exp_e_pos_filtered)}개")
    results['V3_ExpE_POSFiltered'] = exp_e_pos_filtered
    
    # -------------------------------------------------------------------------
    # Variant 4: Domain-Filtered Union (Exp D)
    # -------------------------------------------------------------------------
    print("\n[4] Domain-Filtered Union (Exp D)")
    exp_d_union = exp_a.union(exp_b).union(exp_c)
    exp_d_filtered = filter_with_domain_knowledge(exp_d_union)
    print(f"  원본: {len(exp_d_union)}개 → 필터링: {len(exp_d_filtered)}개")
    results['V4_ExpD_DomainFiltered'] = exp_d_filtered
    
    # -------------------------------------------------------------------------
    # Variant 5: Hybrid (General ∩ Corpus-specific)
    # -------------------------------------------------------------------------
    print("\n[5] Hybrid (General ∩ Domain-Filtered Union)")
    hybrid_intersection = create_hybrid_stopwords(
        exp_d_filtered,
        GENERAL_GRAMMAR_STOPWORDS,
        use_intersection=True
    )
    print(f"  크기: {len(hybrid_intersection)}개")
    results['V5_Hybrid_Intersection'] = hybrid_intersection
    
    # -------------------------------------------------------------------------
    # Variant 6: Conservative (General + Safe Legal Boilerplate)
    # -------------------------------------------------------------------------
    print("\n[6] Conservative (General + 안전한 법률 상투어)")
    conservative = GENERAL_GRAMMAR_STOPWORDS.union(LEGAL_BOILERPLATE_SAFE)
    print(f"  크기: {len(conservative)}개")
    results['V6_Conservative'] = conservative
    
    # -------------------------------------------------------------------------
    # Variant 7: Class-Aware (if corpus and labels provided)
    # -------------------------------------------------------------------------
    if corpus is not None and labels is not None:
        print("\n[7] Class-Aware Stopwords (클래스 판별력 기반)")
        # Exp E를 출발점으로 클래스 판별력 분석
        class_aware = get_class_aware_stopwords(
            exp_e_filtered,
            corpus,
            labels,
            variance_threshold=0.0001
        )
        print(f"  크기: {len(class_aware)}개")
        results['V7_ClassAware'] = class_aware
    
    # -------------------------------------------------------------------------
    # Save all variants
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("💾 Saving improved stopword lists...")
    print("="*70)
    
    for name, stopwords in results.items():
        filepath = os.path.join(output_dir, f"{name}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            for word in sorted(stopwords):
                f.write(word + '\n')
        print(f"  ✅ {name}: {len(stopwords)}개 → {filepath}")
    
    # Summary CSV
    summary = pd.DataFrame([
        {'Variant': name, 'Count': len(words)} 
        for name, words in results.items()
    ])
    summary_path = os.path.join(output_dir, "variants_summary.csv")
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n  📊 Summary: {summary_path}")
    
    return results


# =============================================================================
# 5. USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # 예시: 기존 결과 파일들 로드
    import sys
    
    # CSV 파일 경로 (실제 경로로 수정 필요)
    try:
        df_a = pd.read_csv("stopword_판례_통합/candidates_A_zipf.csv")
        df_b = pd.read_csv("stopword_판례_통합/candidates_B_entropy.csv")
        df_c = pd.read_csv("stopword_판례_통합/candidates_C_mpvp.csv")
        
        set_a = set(df_a['word'].tolist())
        set_b = set(df_b['word'].tolist())
        set_c = set(df_c['word'].tolist())
        
        print(f"✅ Loaded:")
        print(f"  - Exp A (Zipf): {len(set_a)}개")
        print(f"  - Exp B (Entropy): {len(set_b)}개")
        print(f"  - Exp C (MP-VP): {len(set_c)}개")
        
        # Run improvement pipeline
        improved = improve_stopwords(set_a, set_b, set_c)
        
        print("\n" + "="*70)
        print("✨ Improvement Complete!")
        print("="*70)
        print("\n다음 단계:")
        print("1. improved_stopwords/ 폴더의 각 variant로 분류 성능 재평가")
        print("2. 최고 성능 variant 선택")
        print("3. 필요시 수동으로 미세 조정")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n사용법:")
        print("  1. stopword_판례_통합/ 폴더가 있는지 확인")
        print("  2. candidates_A_zipf.csv 등 파일이 있는지 확인")
        print("  3. 경로를 수정하거나 해당 스크립트를 먼저 실행")
