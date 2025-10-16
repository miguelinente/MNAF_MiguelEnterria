#!/usr/bin/env bash
set -euo pipefail

# Fuentes HTTPS que devuelven cabecera Date fiable (puerto 443)
SOURCES=(
  "https://api.github.com"
  "https://www.google.com"
  "https://1.1.1.1"
)

log() { echo "[fix-time] $*"; }

# Si la fecha del sistema está MUY mal (certs no válidos), pon una fecha aproximada válida
# (para que TLS deje de fallar). Ajusta si quieres otra aproximada.
YEAR_NOW=$(date +%Y || echo 1970)
if [ "$YEAR_NOW" -lt 2024 ]; then
  log "Reloj muy atrasado; poniendo fecha aproximada para habilitar TLS..."
  sudo date -s "2025-01-01 00:00:00" || true
fi

# Intenta obtener la hora desde cabecera Date de varias fuentes
for url in "${SOURCES[@]}"; do
  log "Probando $url ..."
  # Extrae la cabecera Date (RFC 7231), ej: 'Date: Fri, 17 Oct 2025 21:43:09 GMT'
  HDR=$(curl -I -s --max-time 5 "$url" | grep -i '^Date:' || true)
  if [ -n "$HDR" ]; then
    # Quita 'Date:' y aplica al sistema
    # shellcheck disable=SC2001
    RFC_DATE=$(echo "$HDR" | sed 's/^[Dd]ate:[ ]*//')
    if [ -n "$RFC_DATE" ]; then
      log "Fijando hora del sistema a: $RFC_DATE"
      sudo date -s "$RFC_DATE" && sudo hwclock -w
      break
    fi
  fi
done

# Si está instalado chrony, fuerza un paso inicial y deja que siga en segundo plano
if systemctl list-unit-files | grep -q '^chrony\.service'; then
  log "Forzando sincronización con Chrony (makestep)..."
  sudo systemctl start chrony || true
  sudo chronyc online || true
  sudo chronyc makestep || true
fi

log "Hecho."
