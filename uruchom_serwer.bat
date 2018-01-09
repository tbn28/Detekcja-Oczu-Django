@echo off

echo ^> makemigrations...
echo.

python manage.py makemigrations stronka

echo.
echo ^> migrate...
echo.

python manage.py migrate

echo.
echo ^> runserver...
echo.

python manage.py runserver

cmd /k
