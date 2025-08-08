#!/bin/bash

# Script para iniciar el servidor Stencil AI
# Autor: Stencil AI Team
# Fecha: 2024

echo "🚀 Iniciando Stencil AI Server..."

# Verificar que estamos en el directorio correcto
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: No se encuentra app/main.py"
    echo "   Asegúrate de estar en el directorio backend/"
    exit 1
fi

# Liberar puerto si está en uso
echo "🔧 Liberando puerto 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Verificar que el módulo se puede importar
echo "✅ Verificando imports..."
python3 -c "import app.main; print('✅ App module imports successfully')" || {
    echo "❌ Error: No se puede importar app.main"
    exit 1
}

# Iniciar servidor
echo "🌐 Iniciando servidor en http://localhost:8000..."
echo "📖 Documentación disponible en http://localhost:8000/docs"
echo "🔄 Presiona Ctrl+C para detener"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 