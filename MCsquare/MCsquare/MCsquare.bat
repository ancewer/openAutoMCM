@echo off
SET This_Dir=%~dp0


C:\Windows\System32\setx MCsquare_Materials_Dir %This_Dir%Materials
start /low /min /wait %This_Dir%MCsquare_win.exe
