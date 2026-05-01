# Session Wrap-up: BM25 법령 검색 실험

작성일: 2026-05-01

## 현재 작업 맥락

연구 주제는 법률 텍스트 검색에서 일반 불용어, 법률 도메인 단일어 불용어, 법률 n-gram 불용어가 BM25 검색 성능에 미치는 영향을 비교하는 것이다.

핵심 실험 파일:

- `06.bm25_statue_retrieval_benchmark.py`
- `06.bm25_statue_retrieval_benchmark_검수후.py`
- `06.bm25_statute_token_ablation.py`

핵심 결과 파일:

- `06.result_bm25_query_statute_matching.csv`
- `06.result_bm25_query_statute_matching_검수후.csv`
- `06.gold_mapping_check_query_statute_matching.csv`

초안 파일:

- `법률 텍스트 검색을 위한 n-gram 기반 도메인 특화 전처리 기법.md`

## 해결한 문제

처음 `06.bm25_statue_retrieval_benchmark.py`를 재실행했을 때 모든 지표가 0으로 나왔다.

원인은 `MANUAL_GOLD_MAP`의 정답 ID가 예전 코퍼스 ID 형식이었다는 점이다. 예전 ID는 대략 다음과 같았다.

```text
docid-8590__dumpdate-20231128___jo-50
```

현재 `statutes_local.jsonl`의 ID는 다음 형식이다.

```text
docid-4209__식품위생법__jo-36
```

따라서 기존 수동 매핑을 그대로 쓰면 BM25가 정답 문서를 찾아도 정답 ID와 매칭되지 않아 평가값이 전부 0이 된다.

## 적용한 수정

`query_statute_matching.jsonl`의 `gold_ref`를 기준으로 현재 코퍼스 ID를 다시 찾도록 수정했다.

추가/수정된 주요 로직:

- `normalize_law_name`
- `build_law_article_index`
- `parse_gold_ref`
- `load_kbl_queries`

매핑 방식:

1. 코퍼스 ID에서 법률명과 조문 번호를 추출한다.
2. `gold_ref`에서 법률명과 조문 번호를 파싱한다.
3. `(정규화된 법률명, 조문번호)`로 현재 코퍼스 ID를 찾는다.

예:

```text
gold_ref: 식품위생법 제36조1항
mapped_id: docid-4209__식품위생법__jo-36
```

가지조문은 현재 코퍼스에서 부모 조문에 합쳐져 있는 경우가 많아 첫 번째 조 번호를 사용한다.

예:

```text
형법 제232-2조 -> jo-232
민사소송법 제202조의2 -> jo-202
```

검증 결과:

```text
[gold 매핑] 성공 52건 / 실패 0건
```

## 생성한 검수 파일

`06.gold_mapping_check_query_statute_matching.csv`를 생성했다.

확인용 컬럼:

- `task_doc_id`
- `query`
- `gold_ref`
- `parsed_law_name`
- `parsed_article_no`
- `mapped_count`
- `mapped_corpus_ids`
- `gold_text_preview`
- `mapped_text_preview`

이 파일을 열면 `gold_ref`가 현재 코퍼스의 어떤 문서 ID로 연결됐는지 사람이 검수할 수 있다.

## 현재 BM25 결과 요약

### 검수전 목록

결과 파일:

```text
06.result_bm25_query_statute_matching.csv
```

가장 좋은 설정:

```text
39. KIL + 법령ngram-050
R@1    0.4038
R@5    0.6346
R@10   0.6923
MRR@10 0.5003
```

Baseline:

```text
00. Baseline (No SW)
R@1    0.3846
R@5    0.6346
R@10   0.6923
MRR@10 0.4868
```

해석:

검수전 목록에서는 n-gram 계열이 상위권을 차지한다. 따라서 “n-gram 기반 전처리가 BM25 법령 검색에서 가장 나은 성능을 보였다”는 주장을 비교적 강하게 할 수 있다.

### 검수후 목록

결과 파일:

```text
06.result_bm25_query_statute_matching_검수후.csv
```

가장 좋은 설정:

```text
07. Baseline + 법령ngram-all
R@1    0.3846
R@5    0.6154
R@10   0.7115
MRR@10 0.4880
```

Baseline:

```text
00. Baseline (No SW)
R@1    0.3846
R@5    0.6346
R@10   0.6923
MRR@10 0.4868
```

General:

```text
01. General (SW)
MRR@10 0.4876
```

해석:

검수후 목록에서도 n-gram 설정이 최고 MRR과 R@10을 기록하긴 하지만, baseline/general과 차이가 매우 작다. 따라서 강한 주장보다는 다음 식으로 쓰는 것이 안전하다.

```text
검수후 목록에서는 n-gram 기반 전처리가 최고 MRR 및 Recall@10을 보였으나,
그 개선 폭은 제한적이었다. 이는 검수 과정에서 과도한 제거 후보가 줄어들면서
검색 성능상의 공격적인 이득도 함께 완화된 결과로 해석된다.
```

## 왜 효과가 약해졌는지에 대한 현재 해석

현재 BM25 실험은 MeCab 명사 기반 토큰화, 즉 `pos.startswith("N")`만 사용한다.

따라서 n-gram 불용어는 BM25에 phrase token으로 들어가는 것이 아니라, 원문에서 구절을 먼저 제거한 뒤 그 안에 포함된 명사들이 함께 제거되는 방식으로 작동한다.

확인한 내용:

- 검수전 ngram-all: 300개
- 검수후 ngram-all: 241개
- 검수전은 더 공격적으로 명사 토큰을 제거한다.
- 검수후는 법률적으로 필요한 표현을 더 보존하므로 성능 개선폭이 줄어든다.

즉, 현재 실험에서 n-gram 효과는 “구절 단위 색인” 효과라기보다 “구절 제거를 통한 명사 토큰 제거 효과”에 가깝다.

## 선행연구 반영 상태

초안 파일:

```text
법률 텍스트 검색을 위한 n-gram 기반 도메인 특화 전처리 기법.md
```

추가한 섹션:

- `2.2.5 한국어 정보검색의 색인 단위와 형태소 선택`
- `3.1.2 데이터 전처리 및 형태소 분석`
- `5.2`의 한계 및 후속 연구 내용 일부

반영한 논지:

- 한국어 정보검색에서는 명사 또는 명사구 기반 색인이 널리 사용된다.
- 한국어는 교착어라 형태소 분석이 검색 성능에 영향을 준다.
- 복합명사/명사구 처리가 중요하다는 선행연구 흐름이 있다.
- 다만 법률 질의에서는 행위, 의무, 금지, 허용을 나타내는 용언도 의미상 중요할 수 있다.
- 따라서 명사 only와 명사+용언 어간 조건을 비교하는 후속 실험이 필요하다.

초안에는 아직 일부 인용이 `[TBD: 선행 연구 인용]` 형태로 남아 있을 수 있다. 다음 세션에서 참고문헌 형식으로 정리하면 좋다.

## 마지막에 만들던 파일

`06.bm25_statute_token_ablation.py`

목적:

BM25 법령 검색에서 형태소 조건 2개를 고정한 뒤, 각 조건에서 불용어 조합 전체를 비교하기 위한 스크립트다.

현재 들어간 기능:

- 검수전/검수후 선택
- 전체 설정/all 또는 대표 설정/core 선택
- 형태소 조건 선택
- 결과 CSV 저장

기본 형태소 조건:

```text
A. 명사 only: N
B. 명사 + 용언 어간: N+VV+VA
```

기본 실행은 모든 목록 버전, 모든 BM25 불용어 설정을 위 두 형태소 조건에서 돈다.

```bash
python 06.bm25_statute_token_ablation.py
```

대표 설정만 빠르게 돌릴 때:

```bash
python 06.bm25_statute_token_ablation.py --config-scope core
```

형태소 조건을 명시해서 돌릴 때:

```bash
python 06.bm25_statute_token_ablation.py --token-modes N,N+VV+VA
```

검수전만:

```bash
python 06.bm25_statute_token_ablation.py --list-version pre
```

검수후만:

```bash
python 06.bm25_statute_token_ablation.py --list-version post
```

스모크 테스트:

```bash
python 06.bm25_statute_token_ablation.py --list-version pre --config-scope core --token-modes N --max-configs 1 --output 06.result_bm25_query_statute_token_ablation_smoke.csv
```

주의:

직전 세션에서 스모크 테스트 실행 중 사용자가 강제 종료했다. 따라서 이 파일은 아직 정상 완료 검증을 끝내지 못했다.

## 다음 세션에서 바로 할 일

1. `06.bm25_statute_token_ablation.py` 문법 확인

```bash
python -m py_compile 06.bm25_statute_token_ablation.py
```

2. 스모크 테스트 실행

```bash
python 06.bm25_statute_token_ablation.py --list-version pre --config-scope core --token-modes N --max-configs 1 --output 06.result_bm25_query_statute_token_ablation_smoke.csv
```

3. 사용자가 원래 원한 두 조건 실행

```bash
python 06.bm25_statute_token_ablation.py --token-modes N,N+VV+VA --config-scope all --output 06.result_bm25_query_statute_token_ablation_N_vs_NVVVA.csv
```

4. 결과 해석 시 볼 포인트

- 검수전/검수후 각각에서 최고 MRR 설정이 무엇인지
- `N`과 `N+VV+VA` 중 어느 쪽이 더 안정적인지
- n-gram 설정이 형태소 조건이 바뀌어도 상위권을 유지하는지
- 용언을 추가했을 때 query token이 증가하지만 noise도 증가하는지
- `EmptyQueryCount`가 특정 형태소 조건에서 증가하는지

## 중요한 해석상 주의점

현재 스크립트의 `VV`, `VA`는 MeCab 품사 태그가 정확히 `VV`, `VA`인 경우만 포함한다.

한국어 용언은 어미와 결합되어 분석되므로, 실제 검색 실험에서 용언 어간을 얼마나 잘 잡는지는 MeCab 분석 결과를 샘플링해서 확인해야 한다.

다음 세션에서 하면 좋은 추가 점검:

```python
MECAB.pos("식품접객업자가 시설기준을 위반하여 영업을 한 경우")
```

이런 식으로 법률 질의 몇 개를 찍어서 `VV`, `VA`가 실제로 어떤 형태로 나오는지 확인해야 한다.

## 현재 결론 초안

현 단계에서 논문에 쓸 수 있는 가장 안전한 결론은 다음과 같다.

```text
검수전 후보군에서는 n-gram 기반 법률 도메인 불용어가 BM25 법령 검색 성능을 가장 크게 개선했다.
검수후 후보군에서도 n-gram 조건이 최고 성능을 보였으나 개선폭은 제한적이었다.
이는 법률 전문가 검수 과정에서 과도한 제거 후보가 줄어들면서,
검색 성능상의 공격적 이득도 함께 완화된 결과로 해석된다.
또한 현재 실험은 명사 기반 색인에 의존하므로,
용언 어간을 포함한 형태소 선택 조건에 따른 추가 검증이 필요하다.
```
