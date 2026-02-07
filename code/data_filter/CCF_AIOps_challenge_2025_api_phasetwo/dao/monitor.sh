#!/bin/bash
# nohup bash monitor.sh &

PID_TO_WATCH=124026
LOG="monitor.log"

echo "[$(date)] 开始监控 PID $PID_TO_WATCH..." >> $LOG

while kill -0 $PID_TO_WATCH 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $PID_TO_WATCH 已结束，启动 dao.sh..." >> $LOG

nohup bash log.sh >> log_dao.log 2>&1 &

echo "[$(date)] dao.sh 已启动，后台 PID: $!" >> $LOG