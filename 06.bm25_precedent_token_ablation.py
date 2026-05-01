import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K_LIST = [1, 5, 10]
POS_GROUPS = ("N", "VV", "VA")
DEFAULT_TOKEN_MODES = ("N", "N+VV+VA")
DEFAULT_OUTPUT = "06.result_bm25_query_precedent_token_ablation_N_vs_NVVVA.csv"
TASKS = (
    ("RELp_Case_to_Case", "case_relevance_p.jsonl"),
    ("RELq_Query_to_Case", "case_relevance_q.jsonl"),
)


def load_module(filename, module_name):
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRE = load_module("06.bm25_precendents_retrieval_benchmark.py", "bm25_precedent_pre")
POST = load_module("06.bm25_precendents_retrieval_benchmark_검수후.py", "bm25_precedent_post")


def init_mecab():
    # 기존 판례 BM25 benchmark와 같은 MeCab 사전 경로를 강제한다.
    # 실패 시 어절 폴백으로 실험값을 만들지 않고 중단해 재현성 위험을 막는다.
    dicpath = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"
    try:
        from konlpy.tag import Mecab

        print(f"  [알림] konlpy.tag.Mecab 사용: {dicpath}")
        return Mecab(dicpath=dicpath)
    except Exception as exc:
        raise RuntimeError(
            "MeCab 초기화에 실패했습니다. 판례 ablation은 반드시 "
            f"Mecab(dicpath='{dicpath}') 조건으로 실행해야 합니다."
        ) from exc


MECAB = init_mecab()


def apply_ngram_stopwords(text, ngrams):
    if not text:
        return ""
    for phrase in ngrams:
        text = text.replace(phrase, " ")
    return text


def parse_token_mode(token_mode):
    groups = tuple(part.strip() for part in token_mode.split("+") if part.strip())
    unknown = [group for group in groups if group not in POS_GROUPS]
    if unknown:
        raise ValueError(f"지원하지 않는 형태소 그룹: {unknown}")
    return groups


def keep_pos(pos, groups):
    pos_parts = set(pos.split("+"))
    return (
        ("N" in groups and pos.startswith("N"))
        or ("VV" in groups and "VV" in pos_parts)
        or ("VA" in groups and "VA" in pos_parts)
    )


def tokenize(text, config, token_mode, apply_word_stopwords=True):
    text = apply_ngram_stopwords(text, config.get("ngrams", []))
    groups = parse_token_mode(token_mode)

    tokens = [word for word, pos in MECAB.pos(text) if keep_pos(pos, groups)]

    if apply_word_stopwords:
        word_sw = config.get("words", set())
        tokens = [tok for tok in tokens if tok not in word_sw]
    return [tok for tok in tokens if len(tok) >= 2]


class BM25Index:
    def __init__(self, corpus_ids, tokenized_corpus, word_stopwords):
        self.ids = corpus_ids
        final_tokenized = [[tok for tok in tokens if tok not in word_stopwords] for tokens in tokenized_corpus]
        self.bm25 = BM25Okapi(final_tokenized)

    def search(self, query_tokens, top_k):
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        return [self.ids[i] for i in np.argsort(scores)[::-1][:top_k]]


def recall_at_k(retrieved_ids, gold_ids, k):
    gold_set = set(gold_ids)
    return 1.0 if any(rid in gold_set for rid in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids, gold_ids):
    gold_set = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate(queries, corpus_ids, base_tokens_list, config, token_mode):
    index = BM25Index(corpus_ids, base_tokens_list, config.get("words", set()))

    recalls = {k: [] for k in TOP_K_LIST}
    reciprocal_ranks = []
    empty_query_count = 0
    evaluated_query_count = 0

    for query in queries:
        gold_ids = query.get("gold_ids", [])
        if not gold_ids:
            continue
        evaluated_query_count += 1
        query_tokens = tokenize(query["query"], config, token_mode, apply_word_stopwords=True)
        if not query_tokens:
            empty_query_count += 1
        retrieved = index.search(query_tokens, top_k=max(TOP_K_LIST))
        for k in TOP_K_LIST:
            recalls[k].append(recall_at_k(retrieved, gold_ids, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved, gold_ids))

    if not evaluated_query_count:
        return {f"Recall@{k}": 0.0 for k in TOP_K_LIST} | {
            "MRR@10": 0.0,
            "EmptyQueryCount": empty_query_count,
            "EvaluatedQueryCount": evaluated_query_count,
        }

    return {
        f"Recall@{k}": round(float(np.mean(recalls[k])), 4)
        for k in TOP_K_LIST
    } | {
        "MRR@10": round(float(np.mean(reciprocal_ranks)), 4),
        "EmptyQueryCount": empty_query_count,
        "EvaluatedQueryCount": evaluated_query_count,
    }


def select_configs(configs, scope):
    if scope == "all":
        return configs

    names = [
        "00. Baseline (No SW)",
        "01. General (SW)",
        "02. KIL Only",
        "07. Baseline + 판례ngram-all",
        "18. KIL + 판례ngram",
        "08-E. Baseline + 판례E",
        "13-E. KIL + 판례E",
        "19-E. Baseline + 판례E + 판례ngram",
        "24-E. KIL + 판례E + 판례ngram",
    ]
    return {name: configs[name] for name in names if name in configs}


def parse_args():
    parser = argparse.ArgumentParser(
        description="BM25 판례 검색에서 N*와 N*+VV+VA 조건별로 불용어 조합을 테스트합니다."
    )
    parser.add_argument(
        "--list-version",
        choices=["pre", "post", "both"],
        default="both",
        help="검수전/검수후 목록 중 어떤 버전을 실행할지 선택합니다.",
    )
    parser.add_argument(
        "--task",
        choices=["relp", "relq", "both"],
        default="both",
        help="RELp(판례-판례), RELq(질의-판례) 중 어떤 태스크를 실행할지 선택합니다.",
    )
    parser.add_argument(
        "--config-scope",
        choices=["all", "core"],
        default="all",
        help="all은 기존 BM25 설정 전체, core는 대표 설정만 실행합니다.",
    )
    parser.add_argument(
        "--token-modes",
        default=",".join(DEFAULT_TOKEN_MODES),
        help="쉼표 구분 목록. 기본값: N,N+VV+VA",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="디버깅용: 앞에서부터 N개 설정만 실행합니다.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="결과 CSV 파일명 또는 절대경로입니다.",
    )
    return parser.parse_args()


def selected_token_modes(raw):
    modes = [mode.strip() for mode in raw.split(",") if mode.strip()]
    for mode in modes:
        parse_token_mode(mode)
    return modes


def selected_modules(list_version):
    modules = []
    if list_version in {"pre", "both"}:
        modules.append(("검수전", PRE))
    if list_version in {"post", "both"}:
        modules.append(("검수후", POST))
    return modules


def selected_tasks(task):
    if task == "relp":
        return TASKS[:1]
    if task == "relq":
        return TASKS[1:]
    return TASKS


def run_suite(list_label, module, task_label, task_file, config_scope, token_modes, max_configs):
    print(f"\n=== {list_label} / {task_label} ===")
    corpus, queries = module.load_case_relevance_task(task_file)
    corpus_ids = [doc["id"] for doc in corpus]
    configs = select_configs(module.build_stopword_configs(module.BASE_DIR), config_scope)

    if max_configs is not None:
        configs = dict(list(configs.items())[:max_configs])

    rows = []
    total = len(token_modes) * len(configs)
    done = 0

    for token_mode in token_modes:
        corpus_cache = {}
        print(f"\n[token_mode={token_mode}]")
        for experiment, config in configs.items():
            done += 1
            ng_tuple = tuple(config.get("ngrams", []))
            cache_key = (token_mode, ng_tuple)
            if cache_key not in corpus_cache:
                temp_config = {"words": set(), "ngrams": list(ng_tuple)}
                corpus_cache[cache_key] = [
                    tokenize(doc["text"], temp_config, token_mode, apply_word_stopwords=False)
                    for doc in tqdm(
                        corpus,
                        desc=f"    corpus tokenize ({token_mode}, ngram={len(ng_tuple)})",
                        leave=False,
                    )
                ]

            print(f"  ({done}/{total}) {experiment}")
            metrics = evaluate(queries, corpus_ids, corpus_cache[cache_key], config, token_mode)
            row = {
                "ListVersion": list_label,
                "Task": task_label,
                "TokenMode": token_mode,
                "Experiment": experiment,
                "WordStopwordCount": len(config.get("words", [])),
                "NgramStopwordCount": len(config.get("ngrams", [])),
            } | metrics
            rows.append(row)
            print(
                f"      R@1={metrics['Recall@1']:.4f} "
                f"R@5={metrics['Recall@5']:.4f} "
                f"R@10={metrics['Recall@10']:.4f} "
                f"MRR={metrics['MRR@10']:.4f} "
                f"empty_q={metrics['EmptyQueryCount']}"
            )
    return rows


def main():
    args = parse_args()
    token_modes = selected_token_modes(args.token_modes)

    print("=" * 78)
    print(" BM25 Query/Case-to-Precedent Tokenization Exhaustive Search")
    print("=" * 78)
    print(f"list_version : {args.list_version}")
    print(f"task         : {args.task}")
    print(f"config_scope : {args.config_scope}")
    print(f"token_modes  : {', '.join(token_modes)}")

    rows = []
    for list_label, module in selected_modules(args.list_version):
        for task_label, task_file in selected_tasks(args.task):
            rows.extend(
                run_suite(
                    list_label,
                    module,
                    task_label,
                    task_file,
                    args.config_scope,
                    token_modes,
                    args.max_configs,
                )
            )

    df = pd.DataFrame(rows).sort_values(
        ["Task", "ListVersion", "MRR@10", "Recall@10", "Recall@5", "Recall@1"],
        ascending=[True, True, False, False, False, False],
    )

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(BASE_DIR, out_path)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")

    print("\n상위 20개")
    print(
        df[
            [
                "ListVersion",
                "Task",
                "TokenMode",
                "Experiment",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "MRR@10",
                "EmptyQueryCount",
                "EvaluatedQueryCount",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
