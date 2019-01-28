# video-mosaic

Simple script to generate video mosaics, where each frame of the output video(s) is composed of tile-sized images or frames from another video.

## How to Use

```
mosaic.py [-h] [--color COLOR] [--blur BLUR] source size out_size

positional arguments:
  source         Tile source directory.
  size           Tile size.
  out_size       Size of output video as tuple in form (rows, columns).

optional arguments:
  -h, --help     show this help message and exit
  --color COLOR  OpenCV color map to apply to tiles.
  --blur BLUR    Apply a Gaussian blur to frames.
```

1. In the `inputs` folder, place the videos you want rendered into mosaics. Note that longer videos will result in longer runtimes so the script in its current form works best on clips of a few seconds long.
2. If the tile source directory contains only a video, the video will be split into frames and the frames saved. If the tile source directory contains any images, these images will be used as tiles and the video file ignored. More images with more intensity/color variation will yield better results. 
3. Tiles are chosen based on the squared difference of mean intensities for each tile region. This is significantly faster than evaluating error on a per-pixel basis, without a meaningful difference in quality of output. Smaller tile sizes will take longer to run but appear more accurate to the original videos.

## Sample Output

![](mosaic.gif)
