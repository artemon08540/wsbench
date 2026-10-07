@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title wsbench
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo Спочатку запустіть install.bat
  pause
  exit /b 1
)

:menu
cls
echo ============================================================
echo   wsbench - порівняння методів збору даних з вебсайтів
echo ============================================================
echo   1  Демо: 4 методи на локальному тестовому сайті (~1 хв)
echo   2  Дашборд: результати демо-прогону
echo   3  Дашборд: синтетичний експеримент (800 прогонів)
echo   4  Дашборд: пілот на реальних сайтах (якщо є results\pilot3)
echo   5  Звіт розділу 3 для синтетичних даних (report.md, tables.xlsx)
echo   6  Модульні тести
echo   7  Перевірити сайти з config\sites.yaml (потрібен інтернет)
echo   8  Довідка по всіх командах
echo   0  Вихід
echo.
set "c="
set /p "c=Ваш вибір: "
if "%c%"=="1" ( "%VPY%" -m wsbench demo --repeats 2 & pause & goto menu )
if "%c%"=="2" ( call :dash results\demo & goto menu )
if "%c%"=="3" ( if not exist results\synthetic\runs.csv "%VPY%" tests\synthetic.py
                call :dash results\synthetic & goto menu )
if "%c%"=="4" ( call :dash results\pilot3 & goto menu )
if "%c%"=="5" ( if not exist results\synthetic\runs.csv "%VPY%" tests\synthetic.py
                "%VPY%" -m wsbench analyze --runs results/synthetic
                start "" "results\synthetic\report"
                pause & goto menu )
if "%c%"=="6" ( "%VPY%" tests\test_core.py & pause & goto menu )
if "%c%"=="7" ( "%VPY%" -m wsbench check-site & pause & goto menu )
if "%c%"=="8" ( "%VPY%" -m wsbench --help & pause & goto menu )
if "%c%"=="0" exit /b 0
goto menu

:dash
if not exist "%~1" (
  echo Немає даних у %~1 - спочатку виконайте відповідний пункт меню.
  pause
  exit /b 0
)
echo Дашборд відкриється в браузері: http://localhost:8501
echo Щоб повернутися в меню - закрийте вкладку і натисніть Ctrl+C тут.
"%VPY%" -m wsbench dashboard --runs %~1
exit /b 0
