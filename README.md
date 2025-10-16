[Unit]
Description=Run fix-time at boot and hourly

[Timer]
OnBootSec=30sec
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
