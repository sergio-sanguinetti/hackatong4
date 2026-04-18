#!/bin/bash
# Jebi Hackathon 2026 - Grupo 04
# Entrypoint: analiza los videos y escribe resultados en ./outputs/
#
# Inputs esperados en ./inputs/:
#   - shovel_left.mp4
#   - shovel_right.mp4
#   - imu_data.csv
#
# Outputs generados en ./outputs/:
#   - analysis.json   Métricas detalladas de ciclos y producción
#   - report.html     Reporte ejecutivo con visualizaciones

set -e

echo "============================================"
echo " Jebi Hackathon 2026 — Grupo 04"
echo " Mining Productivity 2.0 — Hitachi EX-5600"
echo "============================================"
echo ""
echo "Inputs:"
ls -la inputs/
echo ""

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
  echo "[Setup] Instalando dependencias..."
  pip install -r requirements.txt -q
fi

echo "[Run] Ejecutando análisis en modo batch..."
python -m solution.main --batch

echo ""
echo "Outputs generados:"
ls -la outputs/
echo ""
echo "Done."
