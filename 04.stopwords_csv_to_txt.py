import pandas as pd
import os

def convert_csv_to_ngram_txt(csv_path, txt_filename):
    """
    CSV 파일의 'ngram' 컬럼을 읽어 줄바꿈으로 구분된 TXT 파일로 저장합니다.
    """
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return

    try:
        # CSV 로드 (인코딩은 상황에 따라 'utf-8' 또는 'utf-8-sig' 사용)
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        if 'ngram' not in df.columns:
            print(f"❌ '{csv_path}'에 'ngram' 컬럼이 존재하지 않습니다.")
            return

        # n-gram 추출: 결측치 제거 -> 문자열 변환 -> 양끝 공백 제거
        ngrams = df['ngram'].dropna().astype(str).str.strip()
        
        # 중복 제거 및 빈 문자열 제외
        unique_ngrams = [n for n in ngrams.unique() if n]

        # TXT 파일로 저장
        with open(txt_filename, 'w', encoding='utf-8') as f:
            for item in unique_ngrams:
                f.write(f"{item}\n")
        
        print(f"✅ 변환 완료: {txt_filename} (총 {len(unique_ngrams)}개 추출)")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

# ==========================================
# 실행부 (경로를 본인의 환경에 맞게 수정하세요)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 법령 n-gram 변환
convert_csv_to_ngram_txt(
    os.path.join(BASE_DIR, "stopword_ngram_후보", "ngram_candidates_법령.csv"), # 원본 CSV 경로
    "stopwords_ngram_법령.txt"                                              # 생성될 TXT 이름
)

# 2. 판례 n-gram 변환
convert_csv_to_ngram_txt(
    os.path.join(BASE_DIR, "stopword_ngram_후보", "ngram_candidates_판례.csv"), # 원본 CSV 경로
    "stopwords_ngram_판례.txt"                                              # 생성될 TXT 이름
)