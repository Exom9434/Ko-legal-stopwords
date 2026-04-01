import os
import sys
import json

DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
KBL_DIR   = os.path.join(DATA_DIR, "kbl-rag")
TASKS_DIR = os.path.join(KBL_DIR, "tasks")

# =============================================================================
# 헬퍼 함수
# =============================================================================

def pip_install(package: str):
    import subprocess
    print(f"  '{package}' 설치 중...")
    subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"], check=True)

def ensure_package(package: str, import_name: str = None):
    try:
        __import__(import_name or package)
    except ImportError:
        pip_install(package)

def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  저장: {path}  ({len(records)}건)")


# =============================================================================
# Step 1. 코퍼스 다운로드 (기존 유지)
# =============================================================================

def download_corpus():
    corpus_path = os.path.join(KBL_DIR, "corpus", "statutes.jsonl")
    if os.path.exists(corpus_path):
        print(f"[코퍼스] 이미 존재: {corpus_path}  (skip)")
        return

    ensure_package("huggingface_hub")
    from huggingface_hub import snapshot_download

    print(f"[코퍼스] lbox/kbl-rag 다운로드 → {KBL_DIR}")
    snapshot_download(
        repo_id="lbox/kbl-rag",
        repo_type="dataset",
        local_dir=KBL_DIR,
        ignore_patterns=["*.arrow", "*.parquet", ".gitattributes"],
    )
    print("  [완료] 코퍼스 다운로드 성공")


# =============================================================================
# Step 2. 판례 관련성(RELp, RELq) 쿼리 생성 (추가됨)
# =============================================================================

def build_case_relevance_queries(subset_name: str, file_name: str):
    """
    RELp 또는 RELq 데이터를 로드하여 검색 평가용 jsonl로 변환합니다.
    """
    out_path = os.path.join(TASKS_DIR, file_name)
    if os.path.exists(out_path):
        print(f"[쿼리] {file_name} 이미 존재 (skip)")
        return

    ensure_package("datasets")
    from datasets import load_dataset

    print(f"[쿼리] lbox/kbl → {subset_name} 로드 및 변환 중...")
    try:
        ds = load_dataset("lbox/kbl", subset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  [오류] {subset_name} 로드 실패: {e}")
        return

    records = []
    for split in ds:
        for item in ds[split]:
            # RELp와 RELq의 필드명이 미세하게 다를 수 있어 유연하게 처리
            # query_case 관련 텍스트 추출 (요지 + 사실 + 주장 결합)
            q_opinion = item.get("query_case_judicial_opinion") or item.get("query", "")
            q_fact = item.get("query_case_fact", "")
            q_claim = item.get("query_case_claim", "")
            
            # 검색어(Query) 생성
            full_query = f"{q_opinion}\n{q_fact}\n{q_claim}".strip()

            # 대상 판례(Retrieved Case) 관련 텍스트 추출
            r_opinion = item.get("retrieved_case_judicial_opinion", "")
            r_fact = item.get("retrieved_case_fact", "")
            r_claim = item.get("retrieved_case_claim", "")
            
            # 검색 대상 문서(Document) 생성
            full_retrieved = f"{r_opinion}\n{r_fact}\n{r_claim}".strip()

            label = item.get("label") # 'B'가 '예(관련 있음)'
            doc_id = item.get("doc-id") or item.get("doc_id")

            records.append({
                "query": full_query,
                "retrieved_text": full_retrieved,
                "is_relevant": (label == "B"),  # 검색 평가 시 정답 여부 판단용
                "label": label,
                "doc_id": doc_id,
                "meta": item.get("meta", {})
            })

    if records:
        save_jsonl(records, out_path)
    else:
        print(f"  [경고] {subset_name} 변환된 레코드 없음")

# =============================================================================
# Step 2. 쿼리 파일 생성  (lbox/kbl HuggingFace 데이터셋)
# =============================================================================

def build_query_statute_matching():
    """
    lbox/kbl의 kbl_knowledge_query_and_statute_matching_qa 태스크를
    BM25 retrieval 형식으로 변환하여 저장합니다.

    원본 형식 (다지선다):
        question  : "다음 질문에 가장 관련이 깊은 법령을 선택해 주세요. {실제질문}"
        A~E       : "법령명 제N조" 형태의 선택지
        label     : 정답 선택지 키 ("A"~"E")
        label_long: 정답 조문 전문
        doc-id    : corpus docid 번호 (int)

    변환 형식 (retrieval):
        {
          "query":     실제 질문 (보일러플레이트 제거),
          "gold_text": 정답 조문 전문 (label_long),
          "gold_ref":  정답 조문 참조명 ("식품위생법 제36조1항"),
          "doc_id":    corpus docid 번호 (int) — 벤치마크에서 ID 매핑에 사용
        }
    """
    out_path = os.path.join(TASKS_DIR, "query_statute_matching.jsonl")
    if os.path.exists(out_path):
        print(f"[쿼리] query_statute_matching 이미 존재 (skip)")
        return

    ensure_package("datasets")
    from datasets import load_dataset

    print("[쿼리] lbox/kbl → kbl_knowledge_query_and_statute_matching_qa 로드 중...")
    try:
        ds = load_dataset(
            "lbox/kbl",
            "kbl_knowledge_query_and_statute_matching_qa",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [오류] {e}")
        print("  → 이 태스크를 건너뜁니다.")
        return

    # 질문 앞에 붙는 고정 보일러플레이트 제거
    QUESTION_PREFIX = "다음 질문에 가장 관련이 깊은 법령을 선택해 주세요."

    records = []
    for split in ds:
        for item in ds[split]:
            question = item.get("question", "").strip()
            label    = item.get("label", "")
            label_long = item.get("label_long", "")
            doc_id   = item.get("doc-id")          # corpus docid 번호 (int)

            # 보일러플레이트 제거 → 실제 질문만 추출
            query = question
            if query.startswith(QUESTION_PREFIX):
                query = query[len(QUESTION_PREFIX):].strip()

            # 정답 선택지 참조명 (예: "식품위생법 제36조1항")
            gold_ref = item.get(label, "")

            if not query or doc_id is None:
                continue

            records.append({
                "query":     query,
                "gold_text": label_long,
                "gold_ref":  gold_ref,
                "doc_id":    int(doc_id),
            })

    if records:
        save_jsonl(records, out_path)
        print(f"  보일러플레이트 제거 후 질문 예시: {records[0]['query'][:80]}")
    else:
        print("  [경고] 변환된 레코드 없음")


def build_bar_exam_queries():
    """
    lbox/kbl의 변호사 시험 태스크를 retrieval 형식으로 변환합니다.
    여러 bar_exam_* 설정 중 가장 큰 것을 사용합니다.
    """
    out_path = os.path.join(TASKS_DIR, "bar_exam.jsonl")
    if os.path.exists(out_path):
        print(f"[쿼리] bar_exam 이미 존재 (skip)")
        return

    ensure_package("datasets")
    from datasets import load_dataset, get_dataset_config_names

    print("[쿼리] lbox/kbl → bar_exam 설정 목록 확인 중...")
    try:
        configs = get_dataset_config_names("lbox/kbl")
        bar_configs = [c for c in configs if "bar_exam" in c]
        print(f"  발견된 bar_exam 설정: {bar_configs}")
    except Exception as e:
        print(f"  [오류] 설정 목록 조회 실패: {e}")
        return

    if not bar_configs:
        print("  [경고] bar_exam 설정 없음 — 건너뜁니다.")
        return

    records = []
    for cfg in bar_configs:
        try:
            ds = load_dataset("lbox/kbl", cfg, trust_remote_code=True)
        except Exception as e:
            print(f"  [경고] {cfg} 로드 실패: {e}")
            continue
        for split in ds:
            for item in ds[split]:
                question = item.get("question") or item.get("query") or item.get("input", "")
                choices  = item.get("choices", [])
                answer   = item.get("answer")
                # 참조 조문이 있으면 gold_text로 사용
                gold_text = item.get("relevant_statute") or item.get("statute", "")
                if not gold_text and choices and answer is not None:
                    try:
                        gold_text = choices[int(answer)]
                    except (IndexError, ValueError):
                        pass
                if question:
                    records.append({
                        "query":     question,
                        "gold_text": gold_text,
                        "gold_ids":  [],
                    })

    if records:
        save_jsonl(records, out_path)
    else:
        print("  [경고] 변환된 bar_exam 레코드 없음")


# =============================================================================
# Step 3. 구조 출력
# =============================================================================

def print_structure():
    print("\n[다운로드 완료 — 폴더 구조]")
    for root, dirs, files in os.walk(KBL_DIR):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        level  = root.replace(KBL_DIR, "").count(os.sep)
        indent = "  " * (level + 1)
        if level == 0:
            print(f"  kbl-rag/")
        else:
            print(f"{indent}{os.path.basename(root)}/")
        for fname in sorted(files):
            if not fname.startswith("."):
                size_kb = os.path.getsize(os.path.join(root, fname)) // 1024
                print(f"{indent}  {fname}  ({size_kb:,} KB)")


# =============================================================================
# 메인
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print(" KBL 데이터셋 다운로더 (RELp/q 추가 버전)")
    print("=" * 55)

    # 1. 코퍼스
    download_corpus()

    # 2. 기존 RAG용 태스크
    # build_query_statute_matching()
    # build_bar_exam_queries()

    # 3. 신규 판례 관련성 태스크 (RELp, RELq)
    build_case_relevance_queries("kbl_reasoning_case_relevance_qa_p", "case_relevance_p.jsonl")
    build_case_relevance_queries("kbl_reasoning_case_relevance_qa_q", "case_relevance_q.jsonl")

    print("\n[작업 완료] RELp 및 RELq 데이터가 준비되었습니다.")