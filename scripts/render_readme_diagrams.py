"""PNG-схемы для README: блоки ассистента и этапы [0]–[11]."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
FONTS = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = []
    if bold:
        names = [
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path(r"C:\Windows\Fonts\calibrib.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ]
    for p in names + FONTS:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def rounded(draw, box, fill, outline, radius=14, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def multiline(draw, x, y, lines, fnt, fill, lh):
    for i, line in enumerate(lines):
        draw.text((x, y + i * lh), line, font=fnt, fill=fill)
    return y + len(lines) * lh


def arrow_down(draw, x, y0, y1, color="#334155"):
    draw.line((x, y0, x, y1 - 8), fill=color, width=3)
    draw.polygon([(x, y1), (x - 7, y1 - 12), (x + 7, y1 - 12)], fill=color)


def arrow_right(draw, x0, y, x1, color="#334155"):
    draw.line((x0, y, x1 - 8, y), fill=color, width=3)
    draw.polygon([(x1, y), (x1 - 12, y - 7), (x1 - 12, y + 7)], fill=color)


# ── 1. Architecture ──────────────────────────────────────────────────────────

def render_blocks() -> Path:
    W, H = 1600, 1100
    img = Image.new("RGB", (W, H), "#F8FAFC")
    d = ImageDraw.Draw(img)
    title, h2, body, small = font(32, True), font(18, True), font(15), font(13)
    navy, teal, amber, rose, slate, white = (
        "#0F172A",
        "#0F766E",
        "#B45309",
        "#BE123C",
        "#334155",
        "#FFFFFF",
    )

    d.rectangle((0, 0, W, 88), fill=navy)
    d.text((40, 22), "RAG-ассистент АЛ 3У-М6 — блоки системы", font=title, fill=white)
    d.text((40, 58), "Консоль · нормализация · Q-A · кэш · ретривер · генерация", font=small, fill="#94A3B8")

    # User
    rounded(d, (40, 120, 280, 230), "#E0F2FE", "#0284C7", 16)
    center_text(d, 160, 138, "Пользователь", h2, "#0C4A6E")
    for i, t in enumerate(["вопрос в консоли", "история диалога", "команды /qa /trace"]):
        center_text(d, 160, 172 + i * 18, t, small, slate)

    # App
    rounded(d, (340, 120, 620, 230), "#ECFDF5", teal, 16)
    center_text(d, 480, 138, "Точки входа", h2, "#115E59")
    for i, t in enumerate(["python -m assistant_api.app", "python -m assistant_api.batch", "scripts/qa_tool"]):
        center_text(d, 480, 172 + i * 18, t, small, slate)

    arrow_right(d, 280, 175, 340)

    # Pipeline core
    rounded(d, (40, 270, 1560, 760), "#FFFFFF", "#CBD5E1", 18)
    d.text((60, 286), "Ядро пайплайна  RAGPipeline.query()", font=h2, fill=navy)

    blocks = [
        (70, 340, 330, 700, "#EEF2FF", "#4338CA", "Нормализация [0]–[6]",
         ["raw → clean_text → fix_input", "needs_history (без LLM)", "contextualize + validate",
          "canonicalize: intent, слоты", "canonical + cache_key", "DialogState / глоссарий"]),
        (360, 340, 620, 700, "#FDF2F8", rose, "Q-A слой [7]",
         ["qa_base.yaml + индекс", "стоп-правила (МПа, ТО-2)", "direct — ответ из базы",
          "hybrid — ответ в промпт", "none — обычный RAG", "draft-ai → только hybrid"]),
        (650, 340, 910, 700, "#FFF7ED", amber, "Кэш [8]",
         ["SQLite cache__<kb>.db", "1) hash сырого вопроса", "2) cache_key (слоты)",
          "3) семантика + фильтры", "не пишет direct/hybrid", "не пишет «нет в документе»"]),
        (940, 340, 1200, 700, "#F0FDFA", teal, "Ретривер [9]",
         ["Chroma kb__<kb_version>", "поиск по standalone", "RETRIEVER_TOP_K чанков",
          "эмбеддинги OpenAI", "DATA_FILE → чанки", "без [вопрос-ответ]"]),
        (1230, 340, 1490, 700, "#F1F5F9", navy, "Генерация [10]–[11]",
         ["generate_extended_ru.txt", "блоки: Q-A / история", "LLM_MODEL (gpt-4o-mini)",
          "проверка чисел hybrid", "update_state + история", "лог и трассировка"]),
    ]
    for x0, y0, x1, y1, fill, edge, head, lines in blocks:
        rounded(d, (x0, y0, x1, y1), fill, edge, 14)
        d.rectangle((x0, y0, x1, y0 + 48), fill=edge)
        center_text(d, (x0 + x1) / 2, y0 + 14, head, h2, white)
        yy = y0 + 68
        for line in lines:
            d.text((x0 + 18, yy), "•  " + line, font=body, fill=slate)
            yy += 42
    for i in range(4):
        x0 = blocks[i][2]
        x1 = blocks[i + 1][0]
        arrow_right(d, x0, 520, x1)

    # Stores
    stores = [
        (70, 800, 390, 1040, "База знаний", "DATA_FILE — RAG V3.txt\nkb_version = sha256[:12]\nглоссарий из [глоссарий]\n+ glossary_extra.yaml"),
        (430, 800, 750, 1040, "База Q-A", "qa_base.yaml (50 записей)\nqa_tests.yaml (автотесты)\nколлекция qa_verified__*\nqa_tool validate / build"),
        (790, 800, 1110, 1040, "Состояние сессии", "DialogMemory, HISTORY_TURNS\nузлы / вид ТО / операция\npending_input_fix\n/reset очищает историю"),
        (1150, 800, 1490, 1040, "Настройки", ".env  >  env/*.env  >  CLI\nQA_LAYER_ENABLED, QA_MODE\nQA_REQUIRE_VERIFIED\nTRACE_MODE, NORMALIZER_VERSION"),
    ]
    for x0, y0, x1, y1, head, text in stores:
        rounded(d, (x0, y0, x1, y1), "#FFFFFF", "#94A3B8", 12)
        d.text((x0 + 18, y0 + 16), head, font=h2, fill=navy)
        yy = y0 + 52
        for line in text.split("\n"):
            d.text((x0 + 18, yy), line, font=body, fill=slate)
            yy += 26

    path = OUT / "assistant_blocks.png"
    img.save(path, "PNG")
    return path


# ── 2. Pipeline stages ───────────────────────────────────────────────────────

STAGES = [
    ("0", "raw", "Исходный ввод",
     "Строка пользователя сохраняется как raw до конца запроса. Рядом — история последних реплик и DialogState (узел, вид ТО, pending_input_fix).",
     "Не пропускается.", "#0EA5E9", False),
    ("1", "clean_text", "Техническая очистка",
     "NFKC, пробелы, ё→е, кавычки и тире. Единицы: градусов→°C, минут→мин, часов→ч. Варианты ТО (то-1, ТО1) → ТО-1. Регистр в cleaned сохраняется.",
     "Никогда. Повторяется после шага [2].", "#0EA5E9", False),
    ("2", "fix_input", "Раскладка и регистр",
     "homoglyph (латиница внутри русского слова), Caps Lock, раскладка ЙЦУКЕН. ratio ≥ 0.8 — auto; 0.5–0.8 — confirm «Вы имели в виду…?»; ниже — не трогать. Белый список: Mercedes, ATEGO.",
     "INPUT_FIX_ENABLED=false. confirm → ответ пользователю, конец.", "#F59E0B", True),
    ("3", "needs_history", "Нужна ли история",
     "Без LLM, по токенам: 1 пустая история → нет; 2 первый токен а/и/еще/тогда/также; 3 местоимение; 4 нет термина глоссария; 5 иначе нет. «Так какая же трудоемкость ЕО зимой?» → правило 5, False.",
     "NORMALIZATION_ENABLED=false.", "#6366F1", False),
    ("4", "contextualize", "Самостоятельный вопрос",
     "LLM (JSON, temperature=0) переписывает зависимый вопрос. В standalone переносится объект из истории, но не числа из ответа ассистента. «А для СО?» после ТО-1 → «Как часто проводится СО?».",
     "needs_history=False или CONTEXTUALIZATION_ENABLED=false. Ошибка JSON/таймаут → fallback на текст [2].", "#8B5CF6", False),
    ("5", "validate_rewrite", "Проверка переписи",
     "Отклонить, если пропало число/отрицание, утекло число из ответа (answer_leak_number), появился ТО-2, слишком длинный standalone. Сверка с текстом после [2], не с raw.",
     "Пропускается вместе с [4]. needs_clarification → уточнение, конец.", "#8B5CF6", True),
    ("6", "canonicalize", "Слоты и ключ",
     "Intent правилами (safety, periodicity, parameter_value…). Глоссарий + pymorphy3 + rapidfuzz. canonical = intent|node|level|…|residual. cache_key = sha256(NORMALIZER_VERSION|kb_version|canonical).",
     "CANONICALIZATION_ENABLED=false — точный кэш по хешу сырой строки.", "#0F766E", False),
    ("7", "qa_layer", "Проверенные ответы",
     "Стоп-правила (МПа, ТО-2) → none. Поиск по canonical и эмбеддингам paraphrases. direct: близко + margin + тот же intent + запись проверена человеком — без LLM. hybrid (в т.ч. draft-ai) — ответ в промпт, LLM остаётся. none — RAG.",
     "QA_LAYER_ENABLED=false / QA_MODE=off / устаревший kb_version.", "#BE123C", True),
    ("8", "cache", "Кэш ответов RAG",
     "1) query_hash сырого вопроса; 2) cache_key слотов; 3) семантика standalone с фильтрами intent/node/level. Пороги — сходство (≥), не путать с QA_*_MAX_DISTANCE.",
     "hit → [11]. Не ищется и не пишется для direct/hybrid.", "#B45309", True),
    ("9", "retrieval", "Поиск чанков",
     "Chroma коллекция kb__<kb_version>, запрос = standalone (или текст [2], если нормализация выключена). top_k = RETRIEVER_TOP_K (в .env часто 3).",
     "Не выполняется при direct и при попадании в кэш.", "#0284C7", False),
    ("10", "LLM request", "Промпт и генерация",
     "Шаблон generate_extended_ru.txt: фрагменты + опционально проверенный Q-A (hybrid) + история. Модель LLM_MODEL. Hybrid: числа ответа должны быть в Q-A, чанках или вопросе, иначе повтор без блока Q-A.",
     "Не выполняется при direct, cache hit, уточнении.", "#1D4ED8", False),
    ("11", "answer", "Ответ и побочные эффекты",
     "Текст пользователю. update_state. История. Кэш — только обычный RAG (не fallback зависимого вопроса при CACHE_STORE_FALLBACK=false). JSONL лог, блок TRACE_MODE.",
     "Всегда, если запрос не оборвался на confirm/уточнении.", "#0F172A", False),
]


def render_pipeline() -> Path:
    W, margin = 1400, 48
    card_h = 200
    gap = 22
    header = 120
    H = header + margin + len(STAGES) * (card_h + gap) + 40
    img = Image.new("RGB", (W, H), "#F8FAFC")
    d = ImageDraw.Draw(img)
    title, h2, body, small, num = font(30, True), font(18, True), font(15), font(13), font(22, True)
    navy, white, slate = "#0F172A", "#FFFFFF", "#334155"

    d.rectangle((0, 0, W, 100), fill=navy)
    d.text((40, 18), "Обработка запроса — этапы [0]–[11]", font=title, fill=white)
    d.text(
        (40, 60),
        "step_id совпадает в коде, логе и TRACE_MODE. «Пропущен» = шаг не вызывался (не ошибка). Ромб = ранний выход.",
        font=small,
        fill="#94A3B8",
    )

    y = header
    cx_line = 78
    for i, (sid, name, head, desc, skip, color, branch) in enumerate(STAGES):
        y0, y1 = y, y + card_h
        rounded(d, (48, y0, W - 48, y1), white, "#E2E8F0", 14)
        # number badge
        bx0, by0 = 64, y0 + 20
        d.ellipse((bx0, by0, bx0 + 52, by0 + 52), fill=color)
        nw = d.textlength(sid, font=num)
        d.text((bx0 + 26 - nw / 2, by0 + 10), sid, font=num, fill=white)
        d.text((130, y0 + 18), f"[{sid}]  {name}", font=h2, fill=navy)
        d.text((130, y0 + 44), head, font=small, fill=color)

        desc_lines = wrap(d, desc, body, W - 200)
        multiline(d, 130, y0 + 68, desc_lines, body, slate, 20)

        skip_lines = wrap(d, "Пропуск: " + skip, small, W - 200)
        sy = y1 - 18 - 18 * len(skip_lines)
        multiline(d, 130, sy, skip_lines, small, "#64748B", 18)

        if branch:
            d.polygon(
                [(W - 78, y0 + 22), (W - 58, y0 + 42), (W - 78, y0 + 62), (W - 98, y0 + 42)],
                fill="#FEF3C7",
                outline="#D97706",
            )

        if i < len(STAGES) - 1:
            arrow_down(d, cx_line, y1, y1 + gap)
        y = y1 + gap

    path = OUT / "request_pipeline.png"
    img.save(path, "PNG")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = render_blocks()
    b = render_pipeline()
    print(a)
    print(b)


if __name__ == "__main__":
    main()
