@echo off
cd /d "%~dp0dashboard"
call npm.cmd run dev -- --host 127.0.0.1 --port 5173 > "%~dp0runs\dashboard-local.out.log" 2> "%~dp0runs\dashboard-local.err.log"
