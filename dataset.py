import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "corpus.txt"
SEQ_LEN = 7

def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[а-яё]+|[.!?]",
        text.lower()
    )

def prepare_dataset():
    

    text = DATA_PATH.read_text(encoding="utf-8").lower()
    tokens = tokenize(text)

    vocab = sorted(set(tokens))

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    encoded = [word_to_idx[word] for word in tokens]

    

    inputs = []
    targets = []

    for i in range(len(encoded) - SEQ_LEN + 1):
        window = encoded[i:i + SEQ_LEN]

        x = window[:-1]
        y = window[-1]
        inputs.append(x)
        targets.append(y)

    assert len(inputs) == len(targets)
    assert all(
        len(x) == SEQ_LEN - 1
        for x in inputs
    )

    return inputs, targets, word_to_idx, idx_to_word
