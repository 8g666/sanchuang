import re


def tokenize(text: str) -> list[str]:
    """Tokenize English terms and Chinese unigrams/bigrams without external models."""

    normalized = text.lower()
    words = re.findall(r"[a-z0-9_./-]+", normalized)
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return words + chinese_tokens
