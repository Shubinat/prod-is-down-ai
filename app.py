import gradio as gr

from pathlib import Path
from utils import tokenize, detokenize, load_model, DEFAULT_PROMPT, CONTEXT_SIZE, generate


model, word_to_idx, idx_to_word = load_model() # загрузка модели


def chat(
    message: str,
    history: str,
    temperature: float,
    max_new_tokens: int,
    sampling: bool,
) -> str:

    tokens = tokenize(message)

    if not tokens:
        return "Напиши хотя бы одно слово."

    known_tokens = [
        token
        for token in tokens
        if token in word_to_idx
    ]

    # Дополняем короткие сообщения стартовым контекстом (для слишком маленьких сообщений)
    seed_tokens = tokenize(DEFAULT_PROMPT)
    seed_tokens = [
        token for token in seed_tokens
        if token in word_to_idx
    ]

    context_tokens = (
        seed_tokens + known_tokens
    )[-CONTEXT_SIZE:]

    if len(context_tokens) < CONTEXT_SIZE:
        return (
            "Не удалось собрать начальный контекст. "
            "Проверь словарь обученной модели."
        )

    prompt_ids = [
        word_to_idx[token]
        for token in context_tokens
    ]

    generated_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        temperature=temperature,
        max_new_tokens=int(max_new_tokens),
        sampling=sampling,
    )

    generated_words = [
        idx_to_word[token_id]
        for token_id in generated_ids
    ]

    answer = detokenize(generated_words)

    unknown_tokens = [
        token for token in tokens
        if token not in word_to_idx
    ]

    # Показываем, если часть сообщения была проигнорирована
    if unknown_tokens:
        unknown = ", ".join(sorted(set(unknown_tokens)))
        answer += (
            f"\n\n⚠️ Неизвестные токены "
            f"не использованы: {unknown}"
        )

    return answer


with gr.Blocks(title="Prod Is Down AI") as demo:

    gr.Markdown(
        "# 🤖 PROD IS DOWN AI\n"
        "Мини-трансформер на PyTorch.\n\n"
        f"**Словарь:** {len(word_to_idx)} токенов  "
        f"| **Контекст:** {CONTEXT_SIZE} токенов"
    )

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[
            gr.Slider(
                minimum=0.2,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Temperature",
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=40,
                step=1,
                label="Количество новых токенов",
            ),
            gr.Checkbox(
                value=True,
                label="Sampling (случайная генерация)",
            ),
        ],
examples=[
    ["разработчик исправил ошибку в авторизации", 0.8, 40, True],
    ["девопс запустил контейнер и ушёл", 0.8, 40, True],
    ["тестировщик нашёл баг в авторизации", 0.8, 40, True],
    ["тимлид назначил созвон", 0.8, 40, True],
],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
