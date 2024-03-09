@echo off
setlocal enabledelayedexpansion

:: Initialize counter
set COUNT=0

:: Directly parse the output of `py -0`
for /f "tokens=*" %%a in ('py -0') do (
    :: Filter lines that start with a dash, indicating a Python version
    echo %%a | findstr /R "^[ ]*-" > nul && (
        set /a COUNT+=1
        set "PYTHON_VER_!COUNT!=%%a"
        echo !COUNT!. %%a
    )
)

:: Make sure at least one Python version was found
if %COUNT%==0 (
    echo No Python installations found via Python Launcher. Exiting.
    goto end
)

:: Prompt user to select a Python version
set /p PYTHON_SELECTION="Select a Python version by number (default is 1): "
if "!PYTHON_SELECTION!"=="" set PYTHON_SELECTION=1

:: Extract the selected Python version tag and parse the version number more accurately
set SELECTED_PYTHON_VER=!PYTHON_VER_%PYTHON_SELECTION%!

:: The version string is expected to be in the format "-V:X.Y *"
:: We'll use a for loop to extract just the "X.Y" part
for /f "tokens=2 delims=: " %%i in ("!SELECTED_PYTHON_VER!") do (
    set "SELECTED_PYTHON_VER=%%i"
)

:: Confirm the selected Python version
echo Using Python version %SELECTED_PYTHON_VER%

:: Prompt for virtual environment name with default 'venv'
set VENV_NAME=venv
set /p VENV_NAME="Enter the name for your virtual environment (Press Enter for default 'venv'): "

:: Create the virtual environment using the selected Python version
echo Creating virtual environment named %VENV_NAME%...
py -%SELECTED_PYTHON_VER% -m venv %VENV_NAME%

:: Generate the activate_venv.bat file
echo Generating activate_venv.bat...
(
echo @echo off
echo cd %%~dp0
echo set VENV_PATH=%VENV_NAME%
echo.
echo echo Activating virtual environment...
echo call "%%VENV_PATH%%\Scripts\activate"
echo echo Virtual environment activated.
echo cmd /k
) > activate_venv.bat

echo Setup complete. Use 'activate_venv.bat' to activate the virtual environment.

:: Call activate_venv.bat to activate the virtual environment
echo Activating the virtual environment...
call activate_venv.bat

echo Virtual environment %VENV_NAME% is activated. You can reactivate it in the future by running 'activate_venv.bat'.

:: After completing the main part of the script, jump to cleanup
goto cleanup

:cleanup
:: Clean up
echo Cleaning up temporary files...
del temp_py_versions.txt
echo Cleanup complete.

:: End of script
endlocal