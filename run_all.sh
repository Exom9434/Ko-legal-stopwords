#!/bin/bash

# 1. 콘다 설정 로드 (보통 이 경로에 있습니다. 설치 경로가 다르면 수정 필요)
source ~/anaconda3/etc/profile.d/conda.sh
# 미니콘다 사용 시: source ~/miniconda3/etc/profile.d/conda.sh

# 2. 환경 활성화
conda activate nlp_learning

# 3. 파이썬 실행
python3 06.bm25_precendents_retrieval_benchmark.py
python3 06.embedding_retrieval_benchmark.py
python3 07.knn_purity_benchmark.py