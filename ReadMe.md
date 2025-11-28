한국어 법률 텍스트 불용어 목록 구축 및 비교 연구

1. 데이터 수집 (Data Collection)
법령: 국가법령정보센터 API → '법률(Acts)' 전체' 수집
판례: 국가법령정보센터 API → '대법원 판례' 전체 수집비교
평가용 데이터: LBOX Open Benchmark 데이터셋 확보

2. 전처리 및 사전 구축 (Preprocessing)
형태소 분석기: Mecab-ko 사용
사용자 사전: 국가법령정보센터 '법령용어 목록' API 연동하여 Mecab user_dic에 추가품사 필터링: 조사(J), 어미(E), 접사(X) 등 기능어 1차 제거 후 명사(N)/용언(V) 대상으로 실험 진행

3. 불용어 후보 추출 (Methodology Comparison)이 단계에서 두 개의 서로 다른 불용어 후보 리스트를 생성하여 경쟁

기법 A: TF-IDF 기반 (통계적 접근)전체 말뭉치에 대한 단어별 TF-IDF 산출
    가설: DF(문서 빈도)가 너무 높아 IDF 값이 0에 수렴하는 단어들은 변별력이 없는 '범용 법률 용어'이다. 
    Cut-off 기준 설정 (예: 하위 10% IDF 등)

기법 B: 엔트로피(Entropy) 기반 (정보이론적 접근)단어 $t$의 엔트로피 $H(t) = - \sum P(d|t) \log P(d|t)$ 계산
    가설: 엔트로피가 높은(분포가 균일하여 무작위성이 강한) 단어는 특정 판례나 법리를 특정하지 못하는 '잡음'이다.
    Cut-off 기준 설정
    
4. 전문가 정제 (Human-in-the-loop Refinement)
참여자: 변호사 또는 법학 박사 학위 소지자 (최소 1~2인)
작업: 기법 A와 B에서 추출된 후보 리스트(Raw Lists)를 검토 
    기준:"이 단어가 판례의 요지나 법령의 해석을 가르는 핵심 키워드인가?"
    예: '경우', '사항' (삭제 승인) vs '고의', '과실' (삭제 반려 - 법적 요건이므로)
    
    결과물: 정제된 List A(TF-IDF) vs 정제된 List B(Entropy) 도출
    
5. 성능 검증 (Evaluation via LBOX Benchmark)실험 
모델: BERT 기반 법률 모델 (예: KR-BERT-SC, Legal-BERT-KR) 또는 LBOX 벤치마크 베이스라인 모델
실험 방법: 동일한 모델에 전처리 방식만 다르게 하여 성능 비교
Baseline 1: 불용어 제거 안 함 (Full Text)
Baseline 2: 일반 한국어 불용어 리스트 사용
Experiment A: 정제된 List A (TF-IDF) 적용
Experiment B: 정제된 List B (Entropy) 

적용평가 Task (LBOX 데이터셋 활용):
    LCC (Legal Case Classification): 판례 분류 (불용어가 노이즈로 작용하는지 확인)
    LST (Legal Statute Prediction): 관련 법령 예측 (핵심 키워드 보존 여부 확인)
    평가 지표: Accuracy, F1-Score