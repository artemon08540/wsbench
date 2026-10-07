"""Модульні тести ключових функцій (розділ 4.9 роботи). Запуск:  python -m pytest -q  або  python tests/test_core.py"""
from wsbench.collectors.common import extract_from_html, parse_price
from wsbench.domain import PageSpec, ProductRecord
from wsbench.groundtruth import evaluate, normalize_name
from wsbench.harness import plan_runs


def test_parse_price():
    cases = {
        "12 999 ₴": 12999.0, "12 999 грн": 12999.0, "1 299,50 грн": 1299.5,
        "$1,299.99": 1299.99, "1.299,00 zł": 1299.0, "1.299 ₴": 1299.0, "від 999": 999.0,
        "15 499 ₴ 17 999 ₴": 15499.0, "": None, "немає": None, "799.9": 799.9,
    }
    for s, exp in cases.items():
        assert parse_price(s) == exp, (s, parse_price(s), exp)


def test_extract_takes_current_price():
    html = ('<div class="c"><a class="t">Ноутбук A</a><span class="old">20 000 ₴</span><span class="new">18 000 ₴</span></div>'
            '<div class="c"><a class="t">Ноутбук B</a><span class="new">9 999 ₴</span></div>')
    page = PageSpec("p", "s", "u", card_selector="div.c", name_selector=".t", price_selector=".new")
    recs = extract_from_html(html, page)
    assert [(r.name, r.price) for r in recs] == [("Ноутбук A", 18000.0), ("Ноутбук B", 9999.0)]


def test_evaluate_metrics():
    truth = [ProductRecord("Ноутбук Lenovo IdeaPad 3", 20000), ProductRecord("Монітор LG 27\"", 8000),
             ProductRecord("Смартфон Samsung A55", 15000), ProductRecord("Навушники Sony WH-1000", 9000)]
    found = [ProductRecord("НОВИНКА! Ноутбук Lenovo IdeaPad 3", 20000),   # шум у назві – точний збіг після нормалізації
             ProductRecord("Монітор LG 27", 8100),                         # збіг, але ціна неправильна
             ProductRecord("Рекламний банер", 1)]                          # FP
    q = evaluate(found, truth)
    assert (q.tp, q.fp, q.fn) == (2, 1, 2)
    assert q.recall == 50.0 and q.precision == 66.67 and q.field_accuracy == 50.0
    assert abs(q.f1 - 57.14) < 0.01


def test_normalize():
    assert normalize_name("  Ноутбук  ASUS, VivoBook-15 (Хіт) ") == "ноутбук asus vivobook 15"
    assert normalize_name("HP Victus i5\u201113450HX") == normalize_name("HP Victus i5-13450HX")


def test_plan_is_reproducible():
    pages = [PageSpec(f"p{i}", "s", "u", "c", "n", "p") for i in range(5)]
    a = [t.run_id for t in plan_runs(pages, ["M1", "M2", "M3", "M4"], 5, seed=42)]
    b = [t.run_id for t in plan_runs(pages, ["M1", "M2", "M3", "M4"], 5, seed=42)]
    assert a == b and len(a) == 100 and len(set(a)) == 100


def test_holm_and_cliffs():
    from wsbench.analytics.stats import cliffs_delta, delta_label, holm
    assert [round(x, 3) for x in holm([0.01, 0.04, 0.03, 0.5])] == [0.04, 0.09, 0.09, 0.5]
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0 and cliffs_delta([1, 2], [1, 2]) == 0.0
    assert delta_label(0.5) == "великий" and delta_label(0.2) == "малий"


def test_ahp_consistency():
    from wsbench.decision.mcda import ahp
    r = ahp([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert abs(r.weights["K1"] - 1 / 3) < 1e-9 and r.cr < 1e-9
    r = ahp([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])      # свідомо неузгоджені судження
    assert not r.consistent


def test_pareto_and_normalize():
    import pandas as pd
    from wsbench.decision.mcda import normalize, pareto
    t = pd.DataFrame({"recall": [50, 100, 100], "t_ms": [500, 3000, 4000]}, index=["M1", "M3", "M4"])
    _, front = pareto(t)
    assert front == ["M1", "M3"]                       # М4 гірший за М3 за часом і не кращий за повнотою
    n = normalize(t)
    assert n.loc["M1", "t_ms"] == 1.0 and n.loc["M4", "t_ms"] == 0.0 and n.loc["M1", "recall"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
