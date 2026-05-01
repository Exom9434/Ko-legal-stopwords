# 세션 랩업 메모

> 세션 일자: 2026-04-22
> 목적: 이번 세션에서 정리한 논문/프로젝트 작업 내용을 다음 세션에서 바로 이어갈 수 있도록 요약

---

## 1. 이번 세션 핵심 결정

### 프로젝트/논문 방향

- 프로젝트는 `대한민국 법률/판례 텍스트 기반 불용어 구축` 연구로 정리
- 논문 메인 방향은 `법률 retrieval preprocessing` 쪽으로 프레이밍
- 특히 공학 쪽 어필을 위해 `불용어 사전 구축`보다 `phrase-level filtering for sparse legal retrieval`에 초점

### 투고 전략

- 학술대회 proceeding 제출은 **보류**
- proceeding과 저널판 중복성 리스크를 피하기 위해 **저널판 우선 전략**으로 결정
- 저널판은 공학형 포지셔닝으로 다듬되, 현실적인 1차 목표는 `정보처리학회` 계열도 고려

### 본문 범위

- 본문 핵심 실험은 `BM25 + 임베딩` 중심
- `kNN purity`는 본문에서 빼고 **부록/보조 분석**으로 처리

### 전문가 검토 전략

- 변호사 공저자에게는 `n-gram 목록`, `단일어 후보`, `제거하면 안 되는 법률 핵심 표현` 중심 검토 요청
- 검토 결과는 `유지 / 제거 가능 / 애매`로 분류
- 변호사 검토 후에는 **최종 리스트 기준으로 BM25 핵심 실험 재확인** 필요

---

## 2. 이번 세션에서 만든 파일

### 연구설계/논문 관련

- [연구설계.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/연구설계.md)
  - 설계 문서 최신 코드 기준으로 수정
  - `stopwords.txt`를 웹 수집 일반 불용어 비교군으로 명시
  - BM25/kNN/임베딩 설정 번호 체계 최신화
  - 결과표와 출처 CSV 매핑 추가

- [results_summary.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/results_summary.md)
  - 논문/발표용 결과 요약본

- [paper_tables_draft.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/paper_tables_draft.md)
  - 본문 표 1~3 초안
  - 법령 BM25 / 판례 BM25 / 임베딩 비교표 포함

- [appendix_tables_draft.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/appendix_tables_draft.md)
  - 부록 표 구성 초안

### 참고문헌/인용 관련

- [references_draft.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/references_draft.md)
  - 제3장 방법론용 참고문헌 초안
  - LBOX, MeCab, Zipf, Shannon, Kneedle 등 정리

### 전문가 검토 관련

- [lawyer_review_checklist.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/lawyer_review_checklist.md)
  - 변호사 공저자 검토 체크리스트

- [review_after_lawyer_plan.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/review_after_lawyer_plan.md)
  - 변호사 검토 반영 후 재분석 계획

### 심사/논문 방어 관련

- [reviewer_qa_ngram.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/reviewer_qa_ngram.md)
  - bigram/trigram 혼합 설계 방어용 Q&A

### 공학형 프레이밍 관련

- [engineering_framing_checklist.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/engineering_framing_checklist.md)
  - 논문을 공학형 retrieval preprocessing paper로 보이게 하기 위한 장별 체크리스트

- [journal_submission_todo.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/journal_submission_todo.md)
  - 저널판 제출용 우선순위 To-do

### 학술대회 보류 자료

- [kcc_2026_4page_draft.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/kcc_2026_4page_draft.md)
  - KCC 4페이지판 초안 모음
  - 현재는 제출 보류, 추후 발표자료/요약본으로 활용 가능

---

## 3. 이번 세션에서 정리한 논문 본문 방향

### 제1장 서론

- `법률 불용어 구축`보다 `법률 검색에서의 retrieval preprocessing 문제`를 앞세우는 방향으로 리프레이밍
- LLM/RAG 배경은 축소하고 BM25/sparse retrieval 문제의식을 빠르게 제시

### 제2장 관련 연구

- 구조:
  - 법률 코퍼스의 언어적 특성과 전처리 난제
  - 불용어 구축 방법론의 발전
  - 연구 공백과 본 연구의 위치
- 핵심 연결:
  - 단어 기반 접근의 한계
  - 도메인 특화 불용어 연구
  - 법률 텍스트에서 phrase-level filtering 필요성

### 제3장 방법론

- 단일어 후보 추출(TF, Entropy, MP-VP)
- n-gram 후보 추출(bigram/trigram, DF, entropy)
- 법령/판례 분리 코퍼스 설계
- 전문가 검토를 보완 절차로 반영 가능

### 제4장 실험 및 결과

- 본문은 `BM25 + 임베딩` 중심
- `kNN purity`는 부록 이동
- 법령 BM25가 메인 결과
- dense retrieval 결과는 한계가 아니라 `architecture-aware preprocessing insight`로 해석

### 제5장 결론

- `불용어 목록 구축`보다 `법률 sparse retrieval을 위한 phrase-level preprocessing framework`가 핵심 메시지

---

## 4. 변호사 검토 관련 메모

### 변호사에게 꼭 확인받을 것

- 제거하면 안 되는 법률 핵심 표현
- 제거 가능한 전형적 상투 표현
- 법령과 판례에서 다르게 취급해야 하는 표현

### 검토 후 권장 처리

- `검토 후 결과`를 본문 메인으로 사용
- `검토 전/후 비교`는 부록으로 제시
- 분리 기준:
  - `자동 추출 결과`
  - `전문가 검토 반영 결과`

### 재분석 우선순위

1. 법령 BM25 재실행
2. 판례 BM25 재실행
3. 필요 시 임베딩 결과 확인
4. kNN purity는 후순위

---

## 5. 다음 세션 시작 시 가장 먼저 할 일

### 우선순위 1

- 제목 최종안 확정
- 초록 최종본 확정

### 우선순위 2

- 제1장 서론 공학형 최종 정리
- 제2장 관련 연구 최종 문안 정리

### 우선순위 3

- 변호사 공저자 검토 요청 자료 실제 준비
  - 법령 n-gram 상위 30~50개
  - 판례 n-gram 상위 30~50개
  - 단일어 E 후보

---

## 6. 추천 시작 파일

다음 세션에서 바로 열어볼 파일 우선순위:

1. [journal_submission_todo.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/journal_submission_todo.md)
2. [engineering_framing_checklist.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/engineering_framing_checklist.md)
3. [연구설계.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/연구설계.md)
4. [paper_tables_draft.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/paper_tables_draft.md)
5. [lawyer_review_checklist.md](/Users/nojaegyeong/Documents/GitHub/Ko-legal-stopwords/lawyer_review_checklist.md)

---

## 7. 한 줄 요약

```text
이번 세션에서는 논문 방향을 학술대회 proceeding보다 저널판 우선으로 정리했고, 논문을 공학형 legal retrieval preprocessing paper로 포지셔닝하기 위한 문서와 체크리스트를 대부분 준비해두었다.
```
