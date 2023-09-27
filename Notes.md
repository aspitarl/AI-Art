
# Notes 

## Movie Format
Looking at revamping movie file structure to allow for quick indexing of movies to automate playing position in TD Movie In TOP with low CPU/GPU usage. I noticed high CPU usage when moving around the index compared to sequential playing, but later realized it didn't happen all the time.

Made this post 

https://forum.derivative.ca/t/movie-file-in-top-not-interpolating-frames-when-specifying-index/374809

Post about codecs 

https://interactiveimmersive.io/blog/beginner/codecs-for-touchdesigner-explained/

> Photo/Motion JPEG: JPEG encoded video, lossy codec and low decode times with medium file sizes. Good for playback both forwards and backwards or for random access.

command used to make a transition: 

`ffmpeg -framerate 5 -i frame%06d.png -c mjpeg -pix_fmt yuv420p -r 5 output_test.mov`

useful article: https://shotstack.io/learn/use-ffmpeg-to-convert-images-to-video/