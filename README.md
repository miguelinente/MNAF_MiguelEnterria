#!/usr/bin/env bash
set -euo pipefail

# Si la hora es absurda (certs fallan), pon una fecha aproximada para que TLS funcione
YEAR_NOW=$(date +%Y || echo 1970)
if [ "$YEAR_NOW" -lt 2024 ]; then
  sudo date -s "2025-01-01 00:00:00" || true
fi

# Intentar con varias URLs (cualquiera que responda con cabecera Date)
for URL in https://api.github.com https://www.google.com https://1.1.1.1; do
  HDR=$(curl -I -s --max-time 5 "$URL" | grep -i '^Date:' || true)
  if [ -n "$HDR" ]; then
    DATE_STR=$(echo "$HDR" | cut -d' ' -f3-8)
    if [ -n "$DATE_STR" ]; then
      sudo date -s "$DATE_STR" && break
    fi
  fi
done

# Si usas chrony, intenta un ajuste inicial (no es obligatorio)
if systemctl list-unit-files | grep -q '^chrony\.service'; then
  sudo systemctl start chrony || true
  sudo chronyc online || true
  sudo chronyc makestep || true
fi




sudo nano /usr/local/sbin/fix-time.sh
# (pega el script)
sudo chmod +x /usr/local/sbin/fix-time.sh


sudo bash -c 'cat >/etc/systemd/system/fix-time.service << "EOF"
[Unit]
Description=Fix system clock via HTTPS headers, then hand over to chrony
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/env bash -lc "/usr/local/sbin/fix-time.sh"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF'



sudo bash -c 'cat >/etc/systemd/system/fix-time.timer << "EOF"
[Unit]
Description=Run fix-time at boot and hourly

[Timer]
OnBootSec=30sec
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable --now fix-time.timer

sudo systemctl start fix-time.service
chronyc tracking
timedatectl

