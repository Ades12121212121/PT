@echo off
echo 🚀 INSTALANDO PROMPT TRAINER PRO...
echo.

echo 📦 Instalando dependencias...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error instalando dependencias
    echo Intentando instalar manualmente...
    pip install PyQt6 requests numpy pathlib2
)

echo.
echo ✅ Instalación completada!
echo.
echo 📖 Para usar el sistema:
echo 1. Abre LM Studio y carga el modelo Llama 3.2 3B
echo 2. Activa el servidor local en puerto 1234
echo 3. Ejecuta: python prompt_trainer_qt.py
echo.
echo 📚 Lee tutorial.txt para instrucciones detalladas
echo.
pause 