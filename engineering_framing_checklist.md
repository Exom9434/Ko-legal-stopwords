# 공학 어필용 장별 수정 체크리스트

> 목적: 현재 논문을 `불용어 자원 구축 연구`보다 `retrieval preprocessing method`로 읽히게 하기 위한 수정 포인트 정리
> 최종 업데이트: 2026-04-22

---

## 제목

- `불용어 사전 구축`보다 `검색 전처리 기법`, `retrieval framework`, `sparse retrieval optimization` 같은 표현으로 바꾼다.
- `법률 도메인`은 남기되, 응용 분야로 보이게 하고 핵심은 `방법론`에 둔다.
- `n-gram 기반`, `통계 기반`, `BM25`, `전처리` 같은 공학 키워드를 제목에 넣는다.

체크:
- 제목이 `자원 구축 연구`보다 `방법론/시스템 최적화 연구`처럼 읽히는가?
- 제목만 봐도 BM25나 retrieval과 연결되는가?

---

## 1장 서론

- 리걸테크 서비스 사례 나열은 줄이고, `법률 검색 시스템에서 retrieval 성능 개선` 문제를 더 빨리 제시한다.
- LLM/RAG는 배경으로 짧게만 두고, 본 연구의 주 무대는 `BM25 기반 sparse retrieval`임을 초반에 명확히 한다.
- `불용어 처리`를 일반 전처리 문제가 아니라 `retrieval indexing and matching problem`으로 설명한다.
- 법률 도메인의 특수성은 유지하되, `왜 범용 stopword가 retrieval에 문제를 일으키는지`에 초점을 둔다.
- 연구 기여는 `사전 구축`이 아니라 `전처리 프레임워크 제안 + retrieval benchmark 검증`으로 다시 쓴다.

체크:
- 서론 첫 2~3문단 안에 `retrieval`, `BM25`, `preprocessing`, `benchmark`가 등장하는가?
- 연구 목적이 `목록 만들기`보다 `검색 성능 향상 방법 제안`으로 읽히는가?

---

## 2장 관련 연구

- 관련연구를 `불용어 연구 일반론`이 아니라 `retrieval-oriented preprocessing literature`로 정리한다.
- Fox, TF-IDF, entropy 계열 연구를 `index term filtering`, `information content estimation` 관점으로 재서술한다.
- 도메인 특화 연구는 `특정 도메인에서 generic stoplist가 왜 실패하는지`를 보여주는 근거로 사용한다.
- Sarica & Luo 같은 n-gram/technical language 연구는 강하게 살린다.
- 소결에서는 `법률 sparse retrieval에 대한 직접 검증 공백`을 분명하게 적는다.
- 법률 언어 일반론은 너무 길게 가지 않고, 전처리 난제와 직접 연결되는 문장만 남긴다.

체크:
- 각 절이 끝날 때 `그래서 본 연구는 무엇이 다른가`가 드러나는가?
- 관련연구가 survey로 끝나지 않고 `research gap`으로 수렴하는가?

---

## 3장 방법론

- `불용어 추출`보다 `candidate generation`, `scoring`, `adaptive thresholding`, `filtering pipeline` 같은 공학 용어를 더 많이 쓴다.
- 전체 절을 `pipeline` 관점으로 설명한다.
- `단일어 후보 추출 → n-gram 후보 생성 → thresholding → 최종 filtering resource 구성` 흐름을 분명히 한다.
- 법령/판례 분리는 `dataset-specific parameterization`처럼 읽히게 만든다.
- bigram/trigram 혼합은 `practical phrase-level filtering design choice`로 설명한다.
- 수작업 검토는 `human-in-the-loop validation` 또는 `limited expert validation` 정도로 표현을 정리한다.

체크:
- 3장이 자원 구축 서술이 아니라 `알고리즘/파이프라인 설명`처럼 읽히는가?
- 단계별 입력/출력과 판단 기준이 명확한가?

---

## 4장 실험 및 결과

- `실험 환경`을 benchmark-driven evaluation으로 보이게 쓴다.
- BM25와 dense retrieval 비교는 단순 대조가 아니라 `architecture-sensitive preprocessing effect`로 해석한다.
- 단일어 실패는 `unigram filtering limitation`으로 명확히 이름 붙인다.
- n-gram 효과는 `phrase-level filtering improves sparse retrieval`로 요약한다.
- dense retrieval에서 성능 하락은 약점이 아니라 `retrieval architecture dependency`를 보여주는 결과로 해석한다.
- 표 설명은 `최고 성능 나열`보다 `engineering insight`를 강조한다.
- kNN purity는 본문에서 빼고, 논문의 초점을 retrieval evaluation에 고정한다.

체크:
- 결과 해석이 `좋았다/나빴다`가 아니라 `언제 통하고 언제 안 통하는가`를 말하는가?
- BM25 개선이 논문의 메인 엔지니어링 기여로 드러나는가?

---

## 5장 결론

- 결론 첫 문장은 `법률 불용어 목록을 만들었다`보다 `법률 sparse retrieval을 위한 phrase-level preprocessing framework를 제안했다`로 시작하는 게 좋다.
- 기여 요약은 `framework`, `benchmark`, `architecture-aware insight` 중심으로 쓴다.
- dense retrieval 한계는 약점이 아니라 `모델 유형별 전처리 분리 필요성`으로 연결한다.
- 향후 과제는 `hybrid retrieval`, `adaptive preprocessing`, `model-aware filtering`으로 확장한다.
- `속도 향상`, `용량 절감`은 직접 측정하지 않았다면 가능성 수준으로만 언급한다.

체크:
- 결론의 마지막 메시지가 `사전 구축`이 아니라 `retrieval-aware preprocessing`인가?
- 향후 연구가 공학적으로 확장 가능한 방향으로 제시되는가?

---

## 초록

- 첫 문장에서 `법률 검색`과 `BM25 기반 sparse retrieval` 문제를 명확히 제시한다.
- 방법 소개는 `통계 기반 unigram scoring + phrase-level n-gram filtering`처럼 압축한다.
- 결과는 `법령 BM25에서 최고 성능`, `dense retrieval에서는 baseline 우세`를 같이 넣는다.
- 마지막 문장은 `architecture-aware preprocessing strategy` 같은 식으로 마무리한다.

체크:
- 초록만 읽어도 이 논문이 `retrieval preprocessing paper`처럼 보이는가?
- 도메인 응용보다 방법론적 메시지가 먼저 보이는가?

---

## 표/그림

- 표 제목도 공학적으로 바꾼다.
  - `불용어 설정별 성능 비교`보다
  - `전처리 설정별 BM25 성능 비교`
  - `전처리 설정별 dense retrieval 성능 비교`
- 가능하면 파이프라인 그림 1개 추가를 고려한다.
  - `Corpus → Candidate Scoring → Thresholding → n-gram Filtering → Retrieval Evaluation`
- 부록 표는 재현성 자료로 정리하고, 본문 표는 engineering message 중심으로 최소화한다.

체크:
- 표 제목만 봐도 `retrieval experiment`라는 게 드러나는가?
- 그림이 있다면 시스템 흐름이 공학 논문처럼 보이는가?

---

## 용어 통일

- `불용어 사전`과 `불용어 목록`은 필요할 때만 쓰고, 핵심 용어는 `preprocessing resource`, `filtering set`, `retrieval preprocessing` 쪽으로 맞춘다.
- `상투구문`, `관용구`, `형식구`는 하나를 중심 용어로 통일한다.
  - 추천: `상투적 형식구`
- `법률 문서 전처리`보다 `법률 검색 전처리`를 더 자주 쓴다.

체크:
- 같은 개념을 여러 단어로 흔들지 않는가?
- 용어가 정보검색/시스템 논문처럼 정리되어 있는가?

---

## 마지막 자가 점검 질문

- 이 논문은 `법률 불용어 자원 연구`보다 `법률 retrieval preprocessing 연구`처럼 읽히는가?
- 공학 심사자가 봤을 때 `문제 정의-방법-실험-시사점`이 기술 논문 구조로 느껴지는가?
- 도메인은 법률이지만, 메시지는 `검색 시스템 설계 인사이트`로 남는가?
