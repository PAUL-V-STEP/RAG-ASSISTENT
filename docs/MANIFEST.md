# MANIFEST: комплект ТЗ-RAG-AL

Состав комплекта для ветки `assistant_api`. Перед началом работ выполнить `python check_kit.py` (каталог комплекта или `--root <корень проекта>`): он сверяет схемы, счётчики, имена, переменные окружения и контрольные суммы. Файлы с другой sha256 — не из этого комплекта.

| Файл | Место в проекте | Переменная | Содержание | sha256 |
|---|---|---|---|---|
| `ТЗ-RAG-AL.md` | docs/ | — | техническое задание | `b4a502abc6e411867d9c435fcbb2a985de4758d857e275f68b500de79f8df847` |
| `README_RAG-AL.md` | основа для assistant_api/README.md | — | этапы обработки запроса [0]–[11] | `735c7437daacb174c0ea1d683f238f1d2e63588d40ba98a56a27e36820ae3272` |
| `env.example` | корень проекта | — | 96 переменных окружения | `5c67dc2221ffa6a4841094dc73ffb13dba16fdcfd9e713695e795307be224cf0` |
| `ТО автолифта АЛ 3У-М6 — версия для RAG V3.txt` | корень проекта | `DATA_FILE` | база знаний: 38 строк [глоссарий], без [вопрос-ответ] | `024391e66db18304816b4dee08d635d9579400c19a19cae31f9a6e970725ea3e` |
| `qa_base.yaml` | assistant_api/data/qa/ | `QA_BASE_FILE` | schema_version 2: 50 записей, 100 перефразировок, answer_group to1_periodicity | `bb2ded81975f0155f07965656f860b1051f6756eded10a2d83a8adc3e57a0853` |
| `qa_tests.yaml` | assistant_api/data/qa/ | `QA_TESTS_FILE` | schema_version 2: 94 теста (50 match, 26 contrast, 12 no_match, 6 dialog), accept_ids, guard, c13 | `50bdb797cbfe830d74cd12e55a5cfc9a6d17b8c2aec41b94ab93ed3c933d36a5` |
| `batch_example.yaml` | tests/fixtures/ | `BATCH_INPUT_FILE` | пример пакетного прогона, 11 элементов | `2b73a69da5d45913a2312a53d6ca43ccc6af360e5c184c100ce43f8a40e05f72` |
| `check_kit.py` | scripts/ | — | проверка согласованности комплекта | `ffbe032a5c41f7eb3db07d51f57879cb88d167e2b341fcae3698f9a151b78832` |

Не входят в комплект и не индексируются: файл «вопросы и ответы (TXT)» (эталон для оценки), `assistant_api/data/docs.txt`, вариант RAG V3 с приложенным блоком вопросов-ответов.

Устаревшие версии, которые нельзя использовать: `qa_base.yaml` / `qa_tests.yaml` с `schema_version: 1` (92 теста), задание v4 и ТЗ Q-A версии 2 как отдельные документы, имя базы «…версия для RAG V3 без вопросов-ответов (TXT).txt».
