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
echo   ОСНОВНИЙ ЕКСПЕРИМЕНТ (results\main, 22 сторінки)
echo   1  Дашборд результатів
echo   2  Зведені таблиці: за методами і за типом рендерингу
echo   3  Повний звіт розділу 3 (report.md, tables.xlsx, графіки)
echo   4  Перевірити сайти з config\sites_main.yaml (потрібен інтернет)
echo   5  Повторно перевірити API verkkokauppa для M2
echo   6  Запустити експеримент заново (440 прогонів, кілька годин)
echo.
echo   ДЕМО І ТЕСТИ
echo   7  Демо: 4 методи на локальному тестовому сайті (~1 хв)
echo   8  Дашборд: результати демо-прогону
echo   9  Дашборд: синтетичний експеримент (800 прогонів)
echo   10 Звіт розділу 3 для синтетичних даних
echo   11 Модульні тести
echo   12 Довідка по всіх командах
echo   0  Вихід
echo.
set "c="
set /p "c=Ваш вибір: "
if "%c%"=="1" ( call :dash results\main & goto menu )
if "%c%"=="2" goto summary
if "%c%"=="3" goto mainreport
if "%c%"=="4" ( "%VPY%" -m wsbench check-site --sample config/sites_main.yaml --out results/check_main & pause & goto menu )
if "%c%"=="5" goto verk
if "%c%"=="6" goto rerun
if "%c%"=="7" ( "%VPY%" -m wsbench demo --repeats 2 & pause & goto menu )
if "%c%"=="8" ( call :dash results\demo & goto menu )
if "%c%"=="9" ( if not exist results\synthetic\runs.csv "%VPY%" tests\synthetic.py
                call :dash results\synthetic & goto menu )
if "%c%"=="10" ( if not exist results\synthetic\runs.csv "%VPY%" tests\synthetic.py
                "%VPY%" -m wsbench analyze --runs results/synthetic
                start "" "results\synthetic\report"
                pause & goto menu )
if "%c%"=="11" ( "%VPY%" tests\test_core.py & pause & goto menu )
if "%c%"=="12" ( "%VPY%" -m wsbench --help & pause & goto menu )
if "%c%"=="0" exit /b 0
goto menu

:summary
if not exist results\main\runs.csv ( echo Немає results\main\runs.csv & pause & goto menu )
"%VPY%" -c "from wsbench.ui import report as r; df=r.load_runs('results/main/runs.csv'); m=r.method_table(df); t=r.render_table(df); m.to_csv('results/main/summary_by_method.csv', encoding='utf-8-sig'); t.to_csv('results/main/summary_by_render.csv', encoding='utf-8-sig'); print('\nЗа методами:'); print(m.to_string()); print('\nЗа типом рендерингу:'); print(t.to_string()); print('\nЗбережено в results\\main\\summary_by_method.csv і summary_by_render.csv')"
pause
goto menu

:mainreport
if not exist results\main\runs.csv ( echo Немає results\main\runs.csv & pause & goto menu )
"%VPY%" -m wsbench analyze --runs results/main --sample config/sites_main.yaml
start "" "results\main\report"
pause
goto menu

:verk
if not exist check_verk_api.py ( echo Файл check_verk_api.py не знайдено & pause & goto menu )
"%VPY%" check_verk_api.py
pause
goto menu

:rerun
echo.
echo УВАГА: це перезапише results\main (440 прогонів, кілька годин, потрібен інтернет).
set "ok="
set /p "ok=Точно запустити? введіть y: "
if /i not "%ok%"=="y" goto menu
"%VPY%" -m wsbench run-experiment --sample config/sites_main.yaml --methods M1,M2,M3,M4 --repeats 5 --min-delay 10 --out results/main
pause
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
