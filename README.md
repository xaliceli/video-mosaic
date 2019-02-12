# video-mosaic

Simple script to generate video mosaics, where each frame of the output video(s) is composed of tile-sized images or frames from another video.

## How to Use

```
mosaic.py [-h] [--color COLOR] [--blur BLUR] sourceusage: mosaic.py [-h] [--out_size OUT_SIZE] [--color COLOR] [--blur BLUR]
                 source target size vtype

positional arguments:
  source               Tile source directory.
  target               Target source directory.
  size                 Tile size.
  vtype                Mosaic or texture.

optional arguments:
  -h, --help           show this help message and exit
  --out_size OUT_SIZE  Size of output video as tuple in form (rows, columns).
  --color COLOR        OpenCV color map to apply to tiles.
  --blur BLUR          Apply a Gaussian blur to frames. size out_size 
```

1. In the `inputs` folder, place the videos you want rendered into mosaics. 
2. If the tile source directory contains only a video, the video will be split into frames and the frames saved. If the tile source directory contains any images, these images will be used as tiles and the video file ignored. More images with more intensity/color variation will yield better results. 
3. Tiles are chosen based on the squared difference of mean intensities for each tile region. This is significantly faster than evaluating error on a per-pixel basis, without a meaningful difference in quality of output. Smaller tile sizes will take longer to run but appear more accurate to the original videos.

## Sample Output

![](mosaic.gif)
