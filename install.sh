#!/bin/bash

echo "🚀 INSTALANDO PROMPT TRAINER PRO..."
echo

echo "📦 Instalando dependencias..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error instalando dependencias"
    echo "Intentando instalar manualmente..."
    pip3 install PyQt6 requests numpy pathlib2
fi

echo
echo "✅ Instalación completada!"
echo
echo "📖 Para usar el sistema:"
echo "1. Abre LM Studio y carga el modelo Llama 3.2 3B"
echo "2. Activa el servidor local en puerto 1234"
echo "3. Ejecuta: python3 prompt_trainer_qt.py"
echo
echo "📚 Lee tutorial.txt para instrucciones detalladas"
echo 