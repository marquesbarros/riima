@echo off
REM Launches RIIMA with pyw (the windowless Python launcher) so only the app
REM window appears -- no separate console window stays open behind it.
start "" /D "%~dp0" pyw -m riima
