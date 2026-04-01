import os, json, re, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# =============================================================================
# [설정] 경로 및 하이퍼파라미터
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KBL_DIR  = os.path.join(BASE_DIR, "data", "kbl-rag")
MODEL_NAME = "nlpai-lab/KURE-v1"

# 임베딩 결과를 저장할 파일명 (이게 있어야 두 번 고생 안 합니다!)
EMBEDDING_CACHE_PATH = os.path.join(BASE_DIR, "corpus_embeddings_kure_v1.pt")

TOP_K_LIST = [1, 5, 10]

# =============================================================================
# [Part 1] 데이터 로더 (BM25 때 완성한 완벽한 매핑 로직 그대로 사용)
# =============================================================================
def load_kbl_corpus(kbl_dir: str) -> list[dict]:
    path = os.path.join(kbl_dir, "corpus", "statutes.jsonl")
    if not os.path.exists(path): path = os.path.join(kbl_dir, "corpus", "statutes_local.jsonl")
    if not os.path.exists(path): raise FileNotFoundError(f"코퍼스 없음: {path}")

    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj.get("id", ""))
            text = obj.get("contents") or obj.get("text", "")
            m = re.search(r"docid-(\d+)__", doc_id)
            corpus.append({"id": doc_id, "text": text, "docid_num": int(m.group(1)) if m else -1})
    return corpus

def load_kbl_queries(kbl_dir, task, corpus):
    path = os.path.join(kbl_dir, "tasks", f"{task}.jsonl")
    
    def get_char_bigrams(text):
        clean = re.sub(r'[^가-힣0-9]', '', text)
        return set(clean[i:i+2] for i in range(len(clean)-1))

    norm_corpus = [(doc["id"], get_char_bigrams(doc["text"])) for doc in corpus]
    queries = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("query") or obj.get("question") or ""
            gold_text = obj.get("gold_text", "")
            if not q or not gold_text: continue

            gold_bg = get_char_bigrams(gold_text)
            if not gold_bg: continue

            best_id, best_score = None, 0.0
            for doc_id, c_bg in norm_corpus:
                if not c_bg: continue
                score = len(gold_bg & c_bg) / len(gold_bg) # Containment 적용!
                if score > best_score:
                    best_score = score
                    best_id = doc_id

            if best_score > 0.70:
                queries.append({"query": q, "gold_ids": [best_id]})
    return queries

# =============================================================================
# [Part 2] 평가 함수
# =============================================================================
def recall_at_k(retrieved_ids: list, gold_ids: list, k: int) -> float:
    gold_set = set(gold_ids)
    return 1.0 if any(rid in gold_set for rid in retrieved_ids[:k]) else 0.0

def reciprocal_rank(retrieved_ids: list, gold_ids: list) -> float:
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set: return 1.0 / rank
    return 0.0

# =============================================================================
# [Part 3] 메인 실행 (임베딩 캐싱 & 검색)
# =============================================================================
def main():
    print("=" * 70)
    print(f" 🚀 Embedding Retrieval Benchmark ({MODEL_NAME})")
    print("=" * 70)

    # 1. 데이터 로드
    print("[1] 데이터 로드 중...")
    corpus = load_kbl_corpus(KBL_DIR)
    queries = load_kbl_queries(KBL_DIR, "query_statute_matching", corpus)
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [doc["id"] for doc in corpus]

    import gc  # 스크립트 맨 위 import 모여있는 곳에 꼭 추가해 주세요!

# =============================================================================
# (중략... Part 1, Part 2는 그대로 유지)
# =============================================================================

    # 2. 모델 로드 (스마트 하드웨어 감지)
    print(f"\n[2] 모델 로드 중... ({MODEL_NAME})")
    
    # 🔥 어떤 그래픽카드인지 스스로 판단합니다!
    if torch.cuda.is_available():
        device = 'cuda'
        print("  - 🟢 [CUDA] NVIDIA GPU가 감지되었습니다! (초고속 모드)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("  - 🍎 [MPS] Apple Silicon(M칩) GPU가 감지되었습니다!")
    else:
        device = 'cpu'
        print("  - 🐢 [CPU] 사용 가능한 GPU가 없어 CPU로 구동합니다.")
        
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 1024  # OOM 방지를 위한 시퀀스 제한 (필수)


    # 3. 코퍼스 임베딩 (하드웨어 맞춤형 메모리 청소 적용)
    if os.path.exists(EMBEDDING_CACHE_PATH):
        print(f"\n[3] 저장된 코퍼스 임베딩을 불러옵니다: {EMBEDDING_CACHE_PATH}")
        corpus_embeddings = torch.load(EMBEDDING_CACHE_PATH, map_location=device)
    else:
        print(f"\n[3] 코퍼스 전체({len(corpus_texts)}개) 임베딩 변환 시작... (메모리 누수 방지 모드 🛡️)")
        
        all_embeddings = []
        chunk_size = 10000  # 1만 개 단위로 쪼개기
        
        for i in range(0, len(corpus_texts), chunk_size):
            chunk_texts = corpus_texts[i : i + chunk_size]
            current_chunk = (i // chunk_size) + 1
            total_chunks = (len(corpus_texts) - 1) // chunk_size + 1
            
            print(f"\n  ▶️ [Chunk {current_chunk}/{total_chunks}] {i} ~ {i + len(chunk_texts)} 구간 처리 중...")
            
            # NVIDIA 8GB 환경을 고려해 batch_size를 16으로 맞춥니다.
            chunk_emb = model.encode(
                chunk_texts, 
                batch_size=16, 
                show_progress_bar=True, 
                convert_to_tensor=True
            )
            
            # 계산 끝난 벡터를 RAM으로 대피
            chunk_emb = chunk_emb.cpu()
            all_embeddings.append(chunk_emb)
            
            # 🔥 핵심: 현재 사용 중인 기기에 맞춰서 맞춤형 휴지통 비우기!
            if device == 'cuda':
                torch.cuda.empty_cache()  # NVIDIA용 물청소
            elif device == 'mps':
                torch.mps.empty_cache()   # Mac용 물청소
            
            gc.collect()  # 찌꺼기 텍스트 데이터까지 파이썬이 확실하게 청소
                
        # 병합 및 저장
        print("\n  - 모든 청크 병합 중...")
        corpus_embeddings = torch.cat(all_embeddings, dim=0)
        
        torch.save(corpus_embeddings, EMBEDDING_CACHE_PATH)
        print(f"  - 💾 코퍼스 임베딩 100% 저장 완료! ({EMBEDDING_CACHE_PATH})")
        
        corpus_embeddings = corpus_embeddings.to(device)

    # 4. 쿼리 검색 및 평가
    print("\n[4] 쿼리 검색 및 평가 중...")
    recalls = {k: [] for k in TOP_K_LIST}
    rrs = []

    for q in tqdm(queries, desc="  검색 진행도"):
        gold_ids = q["gold_ids"]
        
        # 쿼리를 벡터로 변환
        query_emb = model.encode(q["query"], convert_to_tensor=True)
        
        # 코사인 유사도 계산 (전체 코퍼스 대상 초고속 검색)
        hits = util.semantic_search(query_emb, corpus_embeddings, top_k=max(TOP_K_LIST))[0]
        
        # 검색된 문서 ID 추출
        retrieved_ids = [corpus_ids[hit['corpus_id']] for hit in hits]

        # 평가 지표 기록
        for k in TOP_K_LIST:
            recalls[k].append(recall_at_k(retrieved_ids, gold_ids, k))
        rrs.append(reciprocal_rank(retrieved_ids, gold_ids))

    # 5. 결과 출력
    print("\n" + "=" * 70)
    print(f"🏆 최종 평가 결과 ({MODEL_NAME})")
    print("=" * 70)
    for k in TOP_K_LIST:
        print(f" Recall@{k:<2}: {np.mean(recalls[k]):.4f}")
    print(f" MRR@10   : {np.mean(rrs):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()