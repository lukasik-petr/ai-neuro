@echo off
:: "------------------------------------------------------"
:: " ai-daemon.sh "
:: "------------------------------------------------------"
set FILE_PATH=%cd%
::  Environment Miniconda.
set USID_HOME=C:\Users\%USERNAME%
set CONDA_HOME=%USID_HOME%\miniconda3
set PATH=%PATH%;%CONDA_HOME%\bin
set PYTHONPATH=%CONDA_HOME%\pkgs
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHON_SRC=py-src

set prog=%FILE_PATH%"\ai-daemon.py"
set pidfile=%FILE_PATH%"\pid\ai-daemon.pid"
set logfile=%FILE_PATH%"\pid\ai-daemon.log"

for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set time2=23:59:59
set time1=00:00:01
set date1=2022-01-01
set datestamp=%YYYY%-%MM%-%DD%
set timestamp=%HH%:%Min%:%Sec%
set RETVAL=0
set DEBUG_MODE=debug
set MODEL=DENSE
set EPOCHS=39
set UNITS=71
set LAYERS=2
set BATCH=128
set ACTF=elu
set TXDAT1=%date1%_%time1%
set TXDAT2=%datestamp%_%time2%
set OPTIONS=''
set ILCNT=2
set SHUFFLE=True
set STATUS=run
set curr_timestamp=%datestamp%%timestamp%

call %CONDA_HOME%\Scripts\activate.bat tf

echo ----------------------------------------------------------------
echo ai-daemon.bat 
echo ----------------------------------------------------------------
echo bash: Spusteno s parametry:  
echo    DEBUG_MODE=%DEBUG_MODE%
echo    MODEL=%MODEL%
echo    EPOCHS=%EPOCHS%
echo    BATCH=%BATCH%
echo    UNITS=%UNITS%
echo    LAYERS=%LAYERS%
echo    SHUFFLE=%SHUFFLE%
echo    ACTF=%ACTF%
echo    TXDAT1=%TXDAT1%
echo    TXDAT2=%TXDAT2%
echo    ILCNT=%ILCNT%
echo    STATUS=%STATUS%
echo Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z
echo Start ulohy: %curr_timestamp%
echo Treninkova mnozina v rozsahu: %TXDAT1% : %TXDAT2%
echo ----------------------------------------------------------------
rem eval "$(conda shell.bash hook)"
rem conda_hook.bat
rem conda activate tf
python  .\%PYTHON_SRC%\ai-daemon.py ^
    --status=%STATUS% ^
    --debug_mode=%DEBUG_MODE% ^
    --pidfile=%pidfile%^
    --logfile=%logfile%^
    --model=%MODEL%^
    --epochs=%EPOCHS%^
    --batch=%BATCH%^
    --units=%UNITS%^
    --layers=%LAYERS%^
    --actf=%ACTF%^
    --txdat1=%TXDAT1%^
    --txdat2=%TXDAT2%^
    --ilcnt=%ILCNT%^
    --shuffle=%SHUFFLE%



