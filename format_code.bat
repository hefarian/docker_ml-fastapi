@echo off
REM Script pour formater le code avec Black et isort
REM Usage: format_code.bat

echo ========================================
echo Formatage du code avec Black et isort
echo ========================================

REM Chercher Python
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
    goto :found_python
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py
    goto :found_python
)

echo ERREUR: Python introuvable
echo Installez Python ou ajoutez-le au PATH
pause
exit /b 1

:found_python
echo Python trouve: %PYTHON_CMD%

REM Installer Black et isort si necessaire
echo.
echo [1/3] Installation de Black et isort...
%PYTHON_CMD% -m pip install --quiet black isort

REM Formater avec Black
echo.
echo [2/3] Formatage avec Black (line-length=127)...
%PYTHON_CMD% -m black --line-length 127 app/ tests/

REM Trier les imports avec isort
echo.
echo [3/3] Tri des imports avec isort...
%PYTHON_CMD% -m isort app/ tests/

echo.
echo ========================================
echo Formatage termine avec succes!
echo ========================================
echo.
echo Vous pouvez maintenant commiter les changements:
echo   git add .
echo   git commit -m "Format code with Black and isort"
echo.
pause
