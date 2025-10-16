@echo off

echo Checking for required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Launching Piano.

python functionalPrototype.py

echo.
echo Program finished.
pause
