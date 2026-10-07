@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title wsbench - встановлення
echo.
echo ============================================================
echo   wsbench: встановлення (5-10 хвилин, потрібен інтернет)
echo ============================================================
echo.

rem --- 1. Пошук Python 3.11-3.13 ---------------------------------------
set "PY="
for %%V in (3.12 3.13 3.11) do (
  if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto nopython
echo [1/6] Python знайдено: %PY%

rem --- 2. Віртуальне середовище ---------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [2/6] Створюю віртуальне середовище .venv ...
  %PY% -m venv .venv || goto fail
) else (
  echo [2/6] Віртуальне середовище вже є
)
set "VPY=%~dp0.venv\Scripts\python.exe"

rem --- 3. Бібліотеки ---------------------------------------------------
echo [3/6] Встановлюю бібліотеки ...
"%VPY%" -m pip install --upgrade pip --quiet || goto fail
"%VPY%" -m pip install -r requirements.txt || goto fail
"%VPY%" -m pip install -e . --quiet || goto fail

rem --- 4. Браузер Chromium для методу М3 -------------------------------
echo [4/6] Завантажую браузер Chromium (~150 МБ) ...
"%VPY%" -m playwright install chromium || goto fail

rem Streamlit не питатиме e-mail при першому запуску дашборда
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

rem --- 5. Перевірка ----------------------------------------------------
echo [5/6] Модульні тести ...
"%VPY%" tests\test_core.py || goto fail
echo [6/6] Перевірний прогін на локальному демо-сайті ...
"%VPY%" -m wsbench demo --repeats 1 || goto fail
"%VPY%" tests\synthetic.py >nul
"%VPY%" -m wsbench analyze --runs results/synthetic >nul

echo.
echo ============================================================
echo   ГОТОВО. Для роботи запустіть start.bat
echo ============================================================
pause
exit /b 0

:nopython
echo.
echo Не знайдено Python 3.11-3.13.
echo Встановіть Python 3.12 одним із способів і запустіть install.bat ще раз:
echo   1) https://www.python.org/downloads/release/python-31210/  (поставте галочку "Add python.exe to PATH")
echo   2) у PowerShell:  winget install -e --id Python.Python.3.12
pause
exit /b 1

:fail
echo.
echo *** Помилка на цьому кроці. Скопіюйте текст вище й надішліть авторові. ***
pause
exit /b 1
