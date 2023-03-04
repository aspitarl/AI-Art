cd output
ffmpeg -f concat -safe 0 -i file_order.txt -c copy concatenated.mp4