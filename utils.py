
import re
from pathlib import Path

import torch
from model import MiniTransformer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "out" / "model.pt" # путь к модели

DEFAULT_PROMPT = "бэкендер обсуждал проблему в профиле до самого вечера" # Стартовый контекст из обучающего корпуса
CONTEXT_SIZE = 7 # размер контекста (выбран размер обучения)

def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[а-яёa-z0-9]+|[.!?,:]",
        text.lower(),
    )


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return re.sub(r"\s+([.!?,:])", r"\1", text)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Не найден model.pt. "
            "Сначала запусти python train.py"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    word_to_idx = checkpoint["word_to_idx"]
    idx_to_word = checkpoint["idx_to_word"]

    model = MiniTransformer(
        vocab_size=len(word_to_idx),
        embedding_dim=32,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return model, word_to_idx, idx_to_word

def generate(
    model: MiniTransformer,
    prompt_ids: list[int],
    temperature: float,
    max_new_tokens: int,
    sampling: bool,
) -> list[int]:

    generated = prompt_ids.copy()

    with torch.inference_mode():
        for _ in range(max_new_tokens):

            context = generated[-CONTEXT_SIZE:]

            x = torch.tensor(
                [context],
                dtype=torch.long,
            )

            logits = model(x)

            logits = logits / temperature

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            if sampling:
                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                ).item()
            else:
                next_id = torch.argmax(
                    probabilities,
                    dim=-1,
                ).item()

            generated.append(next_id)

    return generated[len(prompt_ids):]

