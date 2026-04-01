import sys

def process_file(file_path):
    try:
        # 1. 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. ", "를 줄바꿈("\n")으로 변경
        # (혹시 공백이 없는 ","만 바꾸고 싶다면 content.replace(",", "\n")으로 수정하세요)
        new_content = content.replace(", ", "\n")

        # 3. 결과 저장 (원본 파일 이름 앞에 'fixed_'를 붙여 저장합니다)
        output_path = f"fixed_{file_path}"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 변환 완료! '{output_path}' 파일을 확인해보세요.")

    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python replace_text.py [파일명.txt]")
    else:
        process_file(sys.argv[1])