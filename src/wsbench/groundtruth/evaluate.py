"""Еталон (ground truth), нормалізація, зіставлення записів і показники M1–M4 (Recall,
Precision, F1, Field Accuracy).

Формат еталону: ground_truth/<page_id>.csv з колонками  position,name,price
(UTF-8, роздільник – кома). Створюється вручну за процедурою з п. 7.2 ТЗ.
"""
from __future__ import annotations

import csv
import string
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from ..collectors.common import parse_price
from ..domain import ProductRecord

# Слова-шуми, які не допомагають ідентифікувати товар. Список фіксується до експерименту.
NOISE_WORDS = {
    "новинка", "хіт", "акція", "топ", "new", "sale", "hit", "top", "товар", "дня",
    "знижка", "розпродаж", "бестселер", "bestseller", "в", "наявності",
}
# Усі різновиди дефісів/тире (U+2010–U+2015, U+2212) теж вважаються розділовими знаками:
# деякі сайти в DOM замінюють «-» на нерозривний «‑» (U+2011), і без цього назви з М1 і М3 не збігалися б.
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "«»„“”’‘…№\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})
FUZZY_THRESHOLD = 0.92   # мінімальна схожість рядків для нечіткого зіставлення


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е").translate(_PUNCT)
    words = [w for w in s.split() if w not in NOISE_WORDS]
    return " ".join(words)


def read_gt_file(path: Path) -> list[ProductRecord]:
    """Читає CSV еталону з будь-яким роздільником (кома – з програми, крапка з комою – після збереження в
    Excel з польською/українською локаллю) і кодуванням UTF-8 (з BOM або без) чи Windows-1251/1250."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251", "cp1250"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    out = []
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    reader.fieldnames = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
    for i, row in enumerate(reader, start=1):
        name = (row.get("name") or "").strip()
        if not name:
            continue
        pos = row.get("position") or ""
        out.append(ProductRecord(name=name, price=parse_price(row.get("price", "")), price_raw=row.get("price", ""),
                                 position=int(float(pos)) if str(pos).strip().replace(".", "", 1).isdigit() else i))
    return out


def load_ground_truth(gt_dir: Path, page_id: str) -> Optional[list[ProductRecord]]:
    path = Path(gt_dir) / f"{page_id}.csv"
    if not path.exists():
        return None
    return read_gt_file(path)


@dataclass
class Quality:
    tp: int
    fp: int
    fn: int
    recall: float
    precision: float
    f1: float
    field_accuracy: Optional[float]
    n_fuzzy: int            # скільки пар зіставлено нечітко (для перевірки похибки зіставлення)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def match_records(found: list[ProductRecord], truth: list[ProductRecord]) -> list[tuple[int, int, bool]]:
    """Зіставлення один-до-одного. Спочатку точний збіг нормалізованих назв, потім – нечіткий
    (SequenceMatcher ratio >= FUZZY_THRESHOLD), жадібно від найкращої пари.
    Повертає список (index_found, index_truth, fuzzy?)."""
    nf = [normalize_name(r.name) for r in found]
    nt = [normalize_name(r.name) for r in truth]
    free_t: dict[str, list[int]] = {}
    for j, k in enumerate(nt):
        free_t.setdefault(k, []).append(j)

    pairs: list[tuple[int, int, bool]] = []
    used_f, used_t = set(), set()
    for i, k in enumerate(nf):
        if free_t.get(k):
            j = free_t[k].pop(0)
            pairs.append((i, j, False))
            used_f.add(i)
            used_t.add(j)

    rest_f = [i for i in range(len(found)) if i not in used_f]
    rest_t = [j for j in range(len(truth)) if j not in used_t]
    if rest_f and rest_t:
        cand = []
        for i in rest_f:
            for j in rest_t:
                r = SequenceMatcher(None, nf[i], nt[j]).ratio()
                if r >= FUZZY_THRESHOLD:
                    cand.append((r, i, j))
        for _, i, j in sorted(cand, reverse=True):
            if i in used_f or j in used_t:
                continue
            pairs.append((i, j, True))
            used_f.add(i)
            used_t.add(j)
    return pairs


def evaluate(found: list[ProductRecord], truth: list[ProductRecord], price_tol: float = 0.01) -> Quality:
    pairs = match_records(found, truth)
    tp = len(pairs)
    fp = len(found) - tp
    fn = len(truth) - tp
    recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    if tp:
        def same(a, b):
            # обидві ціни відсутні (товар «Продано» без ціни) – теж правильний результат
            if a is None and b is None:
                return True
            return a is not None and b is not None and abs(a - b) <= price_tol
        ok = sum(1 for i, j, _ in pairs if same(found[i].price, truth[j].price))
        fa = 100.0 * ok / tp
    else:
        fa = None
    return Quality(tp, fp, fn, round(recall, 2), round(precision, 2), round(f1, 2),
                   None if fa is None else round(fa, 2), sum(1 for p in pairs if p[2]))


def write_ground_truth_template(path: Path, records: list[ProductRecord]) -> None:
    """Чернетка еталону з результату збору – ЛИШЕ як заготовка для ручної перевірки.
    Кожен рядок треба звірити зі сторінкою в браузері; інакше еталон не є незалежним."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 з BOM і роздільник «;» – так файл одразу правильно відкривається в Excel (PL/UA локаль).
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["position", "name", "price", "note"])
        for r in records:
            price = "" if r.price is None else (f"{r.price:.2f}".replace(".", ","))
            w.writerow([r.position, r.name, price, ""])


def agreement(first: list[ProductRecord], second: list[ProductRecord], price_tol: float = 0.01) -> dict:
    """Надійність еталону (п. 7.2 ТЗ): збіг двох незалежних розміток однієї сторінки.
    Збіг = запис є в обох розмітках (за нормалізованою назвою) І ціни однакові.
    Частка збігу = збіги / кількість унікальних записів в обох розмітках (суворий коефіцієнт Жаккара)."""
    pairs = match_records(first, second)
    same_price = sum(1 for i, j, _ in pairs if first[i].price is not None and second[j].price is not None
                     and abs(first[i].price - second[j].price) <= price_tol)
    union = len(first) + len(second) - len(pairs)
    only_first = [first[i].name for i in set(range(len(first))) - {i for i, _, _ in pairs}]
    only_second = [second[j].name for j in set(range(len(second))) - {j for _, j, _ in pairs}]
    price_diff = [(first[i].name, first[i].price, second[j].price) for i, j, _ in pairs
                  if not (first[i].price is not None and second[j].price is not None and abs(first[i].price - second[j].price) <= price_tol)]
    return {"записів_1": len(first), "записів_2": len(second), "спільних назв": len(pairs), "збіг назва+ціна": same_price,
            "збіг, %": round(100 * same_price / union, 2) if union else 100.0,
            "лише в 1-й": only_first, "лише в 2-й": only_second, "різні ціни": price_diff}

