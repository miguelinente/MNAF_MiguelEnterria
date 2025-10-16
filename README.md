which hwclock || ls -l /usr/sbin/hwclock /sbin/hwclock 2>/dev/null
ls -l /dev/rtc* 2>/dev/null || true
ls /sys/class/rtc 2>/dev/null || true
dmesg | grep -i rtc | tail -n +1
if command -v hwclock >/dev/null 2>&1 && [ -e /dev/rtc ] ; then
  sudo hwclock -w || true
fi
rtcsync
makestep 1.0 3
sudo systemctl restart chrony
sudo chronyc online
sudo chronyc makestep
sudo find / -type f -name hwclock 2>/dev/null
ls -l /dev/rtc* ; ls /sys/class/rtc ; which hwclock
