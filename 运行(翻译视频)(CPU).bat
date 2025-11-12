@echo off
chcp 65001
set cpath=%~dp0
set cpath=%cpath:~0,-1%
"%cpath%\infer.exe" --audio_suffixes="mp4,mkv,avi,mov,webm,flv,wmv" --sub_formats="srt,vtt,lrc" --device="cpu" %*
pause
