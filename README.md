# PROD IS DOWN AI 🤖

Мини-трансформер, который генерирует странные истории из быта айтиспециалистов.

## Запуск по порядку
Данные для обучения + сама модель в гит не загружена, для запуска требуется:
1. Сгенерировать данные для обучения (`build_corpus.py`).
2. Запустить обучение модели.
3. Запустить `app.py`.

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_corpus.py
python train.py
python app.py
```
### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_corpus.py
python train.py
python app.py
```

