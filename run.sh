#!/bin/bash

set -euo pipefail

echo "Jebi Hackathon 2026 - Grupo 04"
echo "Instalando dependencias..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Ejecutando procesamiento batch..."
python -m solution.main --batch

echo "Outputs generados:"
ls -la outputs/
echo "Done."
