# Detener cualquier servidor previo
# Nota: Windows Store Python corre como python3.11/py, no solo python
Stop-Process -Name python,python3.11,py -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Limpiar variables de entorno
Remove-Item Env:GMGN_MOCK -ErrorAction SilentlyContinue

# Configurar API Key y arrancar en modo Live
$env:GMGN_API_KEY="gmgn_7565779d3cd1c4d6cb81c22adb74060e"
Write-Host "Arrancando servidor en modo LIVE en http://127.0.0.1:8000..."
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
