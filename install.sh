#!/usr/bin/env bash
# wsbench: встановлення для macOS / Linux.  Запуск:  bash install.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for v in python3.12 python3.13 python3.11 python3; do
  if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)'; then PY="$v"; break; fi
done
[ -z "$PY" ] && { echo "Потрібен Python 3.11-3.13 (https://www.python.org/downloads/)"; exit 1; }
echo "[1/5] Python: $PY"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python
echo "[2/5] Бібліотеки"; "$VPY" -m pip install --upgrade pip -q; "$VPY" -m pip install -r requirements.txt; "$VPY" -m pip install -e . -q
echo "[3/5] Chromium";   "$VPY" -m playwright install chromium
mkdir -p ~/.streamlit; [ -f ~/.streamlit/credentials.toml ] || printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml
echo "[4/5] Тести";      "$VPY" tests/test_core.py
echo "[5/5] Демо";       "$VPY" -m wsbench demo --repeats 1
"$VPY" tests/synthetic.py >/dev/null; "$VPY" -m wsbench analyze --runs results/synthetic >/dev/null
echo "ГОТОВО. Запуск:  source .venv/bin/activate && python -m wsbench --help"
