@echo off
rem 공정 설비 데이터 분석 - 이 파일을 더블클릭하면 실행됩니다.
rem 처음 실행: 가상환경(.venv)을 만들고 라이브러리를 설치합니다. 몇 분 걸리고 인터넷이 필요합니다.
rem 그다음부터: requirements.txt 가 바뀌었을 때만 다시 설치하고 바로 실행합니다.
cd /d "%~dp0"
title 공정 설비 데이터 분석
set "PY=.venv\Scripts\python.exe"
set "CREATED="

if exist "%PY%" goto check_version
echo.
echo [처음 실행] 가상환경을 만듭니다...
for %%V in (3.12 3.13 3.11) do if not exist "%PY%" py -%%V -m venv .venv >nul 2>&1
if not exist "%PY%" python -m venv .venv
if not exist "%PY%" goto no_python
set "CREATED=1"

:check_version
"%PY%" -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)"
if errorlevel 1 goto bad_version

fc /b requirements.txt ".venv\requirements.installed" >nul 2>&1
if not errorlevel 1 goto run
echo.
echo 라이브러리를 설치하거나 업데이트합니다. 처음에는 몇 분 걸리니 창을 닫지 마세요...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed
copy /y requirements.txt ".venv\requirements.installed" >nul

:run
echo.
echo 프로그램을 시작합니다. 잠시 후 브라우저가 열립니다.
echo 이 검은 창을 닫으면 프로그램도 꺼집니다. 쓰는 동안에는 닫지 마세요.
echo.
"%PY%" -m streamlit run app.py
echo.
echo 프로그램이 멈췄습니다. 위의 메시지를 확인하세요.
pause
exit /b

:no_python
echo.
echo Python 을 찾을 수 없습니다.
echo python.org 에서 Python 3.12 를 설치하세요. 설치 첫 화면에서 "Add python.exe to PATH" 를 꼭 체크하세요.
pause
exit /b 1

:bad_version
echo.
echo 이 Python 버전으로는 실행할 수 없습니다. 3.11 ~ 3.13 이 필요합니다 - 3.12 권장.
"%PY%" --version
if defined CREATED rmdir /s /q .venv
if not defined CREATED echo 이 폴더의 .venv 폴더를 지운 뒤 다시 실행하세요.
echo python.org 에서 Python 3.12 를 설치한 뒤 이 파일을 다시 더블클릭하세요.
pause
exit /b 1

:pip_failed
echo.
echo 라이브러리 설치에 실패했습니다. 인터넷 연결과 회사 프록시 설정을 확인하세요 - 매뉴얼 1.2, 1.5.
pause
exit /b 1
