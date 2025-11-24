@echo off
chcp 65001
set cpath=%~dp0
set cpath=%cpath:~0,-1%
echo ========================================
echo   GPU批处理加速模式 (Batch Inference)
echo   自动检测最佳批处理大小以提高速度
echo   需要更多显存 (建议8GB+)
echo ========================================
"%cpath%\infer.exe" --audio_suffixes="mp3,wav,flac,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv" --sub_formats="srt,vtt,lrc" --device="cuda" --enable_batching --max_batch_size=8 %*
pause