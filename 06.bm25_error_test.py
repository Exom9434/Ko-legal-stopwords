import json
import os

KBL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kbl-rag")
task_file = os.path.join(KBL_DIR, "tasks", "query_statute_matching.jsonl")

print("\n🔍 [긴급 진단] 쿼리 파일의 실제 JSON 구조 확인 🔍")
with open(task_file, "r", encoding="utf-8") as f:
    first_line = f.readline()
    data = json.loads(first_line)
    
    print("\n[첫 번째 문제의 전체 데이터]")
    for key, value in data.items():
        print(f" - {key}: {value}")