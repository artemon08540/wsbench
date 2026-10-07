"""Генерує локальний демо-«магазин» із трьома сторінками різного типу рендерингу та еталон.

  ssr.html     – усі 24 картки вже є в HTML (серверний рендеринг);
  spa.html     – HTML порожній, 36 товарів малює JavaScript із api/spa.json,
                 показує по 12 і довантажує решту при прокручуванні (нескінченний скрол);
  hybrid.html  – 10 карток у HTML, ще 20 довантажує JavaScript з api/hybrid.json.

У кожній картці є стара (закреслена) і нова ціна – щоб перевірити, що селектор бере актуальну.
Запуск:  python demo/build_demo.py
"""
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
GT = ROOT / "ground_truth"

BRANDS = ["Lenovo", "ASUS", "Acer", "HP", "Dell", "Samsung", "Xiaomi", "Apple", "MSI", "Huawei"]
KINDS = ["Ноутбук", "Монітор", "Смартфон", "Планшет", "Навушники", "Роутер"]


def products(n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    out = []
    for i in range(1, n + 1):
        price = rnd.randrange(999, 89999, 10)
        out.append({
            "id": seed * 1000 + i,
            "name": f"{rnd.choice(KINDS)} {rnd.choice(BRANDS)} {rnd.choice('ABCDEFGXZ')}{rnd.randint(100, 9999)} {rnd.choice(['8/256 ГБ', '16/512 ГБ', '27\"', 'Black', 'Silver', 'Pro'])}",
            "price": price,
            "old_price": price + rnd.randrange(200, 5000, 10),
        })
    return out


def fmt(p: int) -> str:
    return f"{p:,}".replace(",", " ") + " ₴"


def card_html(p: dict) -> str:
    return (f'<div class="product-card" data-id="{p["id"]}">'
            f'<a class="product-card__title" href="/p/{p["id"]}">{p["name"]}</a>'
            f'<div class="product-card__prices"><span class="price-old">{fmt(p["old_price"])}</span>'
            f'<span class="price-new">{fmt(p["price"])}</span></div></div>')


HEAD = """<!doctype html><html lang="uk"><head><meta charset="utf-8"><title>{t}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.product-card{{border:1px solid #ddd;padding:10px;min-height:260px}} .price-old{{text-decoration:line-through;color:#999;margin-right:8px}}</style>
</head><body><h1>{t}</h1><nav class="menu"><a>Акції</a> <a>Ноутбуки</a></nav>"""

RENDER_JS = """
<script>
function fmt(p){return p.toLocaleString('uk-UA').replace(/\\s/g,'\\u00a0')+' ₴'}
function card(p){return `<div class="product-card" data-id="${p.id}"><a class="product-card__title" href="/p/${p.id}">${p.name}</a>`+
 `<div class="product-card__prices"><span class="price-old">${fmt(p.old_price)}</span><span class="price-new">${fmt(p.price)}</span></div></div>`}
</script>"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_gt(page_id: str, items: list[dict]) -> None:
    GT.mkdir(parents=True, exist_ok=True)
    with (GT / f"{page_id}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "name", "price"])
        for i, p in enumerate(items, start=1):
            w.writerow([i, p["name"], p["price"]])


def main() -> None:
    ssr = products(24, 1)
    write(SITE / "ssr.html", HEAD.format(t="Ноутбуки (SSR)") + '<div class="grid">' + "".join(map(card_html, ssr))
          + '</div><div class="banner"><span class="price-new">999 ₴</span> Рекламний банер</div></body></html>')
    write_gt("demo_ssr", ssr)

    spa = products(36, 2)
    write(SITE / "api" / "spa.json", json.dumps({"data": {"total": len(spa), "products": spa}}, ensure_ascii=False))
    write(SITE / "spa.html", HEAD.format(t="Смартфони (SPA)") + '<div id="app" class="grid"></div>' + RENDER_JS + """
<script>
let all=[], shown=0;
function more(){const next=all.slice(shown, shown+12); shown+=next.length;
  document.getElementById('app').insertAdjacentHTML('beforeend', next.map(card).join(''));}
setTimeout(()=>fetch('api/spa.json').then(r=>r.json()).then(j=>{all=j.data.products; more();}), 300);
window.addEventListener('scroll',()=>{ if(window.innerHeight+window.scrollY>=document.body.scrollHeight-50 && shown<all.length) setTimeout(more,200); });
</script></body></html>""")
    write_gt("demo_spa", spa)

    hyb = products(30, 3)
    write(SITE / "api" / "hybrid.json", json.dumps({"items": hyb[10:]}, ensure_ascii=False))
    write(SITE / "hybrid.html", HEAD.format(t="Монітори (гібрид)") + '<div id="list" class="grid">'
          + "".join(map(card_html, hyb[:10])) + "</div>" + RENDER_JS + """
<script>
fetch('api/hybrid.json').then(r=>r.json()).then(j=>{document.getElementById('list').insertAdjacentHTML('beforeend', j.items.map(card).join(''))});
</script></body></html>""")
    write_gt("demo_hybrid", hyb)
    print("Демо-сайт і еталон згенеровано:", SITE, GT)


if __name__ == "__main__":
    main()
