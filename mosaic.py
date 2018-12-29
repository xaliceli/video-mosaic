"""
mosaic.py
Creates video mosaics frame-by-frame.
"""
from glob import glob
import os
import sys

import cv2
import numpy as np

def resize_image(image, dims):
    """
    Resizes image to supplied dimensions, preserving aspect ratio.
    """
    row_factor, col_factor = 1, 1
    if image.shape[0] > dims[0]:
        row_factor = float(dims[0]) / image.shape[0]
    if image.shape[1] > dims[1]:
        col_factor = float(dims[1]) / image.shape[1]
    factor = max(row_factor, col_factor)
    if factor < 1:
        image = cv2.resize(image, fx=factor, fy=factor, dsize=None, interpolation=cv2.INTER_AREA)
    resized = image[(image.shape[0] - dims[0])/2:(image.shape[0] - dims[0])/2 + dims[0],
                    (image.shape[1] - dims[1])/2:(image.shape[1] - dims[1])/2 + dims[1]]
    return resized

class VideoMosaic():
    """
    Generates video mosaics.
    """
    def __init__(self, source_dir='input/tiles', size=25, frame_int=5000):
        self.frames = None
        self.tiles = None
        self.out = None
        self.tile_size = size
        self.init_source(source_dir, frame_int)

    def init_target(self, video, dims):
        """
        Splits target video into frames and stores frames in numpy array.
        """
        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        count = 0
        while success:
            print('Reading target frame ' + str(count))
            image = np.array([resize_image(image, dims)])
            self.frames = np.concatenate((self.frames, image), axis=0)
            success, image = vidcap.read()
            count += 1

    def init_source(self, source_dir, frame_int):
        """
        Reads in source images, resizes into square tile, and stores in numpy array.
        """
        if self.tiles is None:
            self.tiles = np.zeros((0, self.tile_size, self.tile_size, 3))
        extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                      'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']
        search_paths = [os.path.join(source_dir, '*.' + ext) for ext in extensions]
        image_files = sorted(sum(map(glob, search_paths), []))

        if image_files:
            images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

            for num, image in enumerate(images):
                print('Reading source frame ' + str(num))
                image = np.array([resize_image(image, (self.tile_size, self.tile_size))])
                self.tiles = np.concatenate((self.tiles, image), axis=0)
        else:
            vidcap = cv2.VideoCapture(os.path.join(source_dir, 'source.mp4'))
            success, image = vidcap.read()
            count = 0
            while success:
                print('Reading source frame from video ' + str(count))
                image = image[10:image.shape[0] - 10, ...]
                if np.all(np.std(image, axis=(0, 1)) > 1):
                    cv2.imwrite(os.path.join(source_dir, 'frame{0:04d}.jpg'.format(count/frame_int)), image)
                    image = np.array([resize_image(image, (self.tile_size, self.tile_size))])
                    self.tiles = np.concatenate((self.tiles, image), axis=0)
                count += frame_int
                vidcap.set(cv2.CAP_PROP_POS_MSEC, count)
                success, image = vidcap.read()

    def good_match(self, region, threshold=.1):
        """
        Returns best match index for tile region.
        """
        diff = np.power(self.tiles - region, 2)
        total_diff = np.sum(diff, axis=(1, 2, 3))
        match_values = np.where(total_diff <= np.min(total_diff) * (1 + threshold))
        good = np.random.choice(match_values[0])
        return good

    def generate_mosaic(self, target, dims, out, frate=24):
        """
        Generates and writes video mosaic.
        """
        if self.frames is None:
            self.frames = np.zeros((0, dims[0], dims[1], 3))
            self.init_target(target, dims)
        self.out = np.zeros(self.frames.shape)

        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (dims[1], dims[0]))
        for fnum, frame in enumerate(self.frames):
            print('Generating output frame ' + str(fnum))
            for row in range(self.tile_size, dims[0], self.tile_size):
                for col in range(self.tile_size, dims[1] + 1, self.tile_size):
                    region = frame[row - self.tile_size:row, col - self.tile_size:col, :]
                    best = self.tiles[self.good_match(region), ...]
                    self.out[fnum, row - self.tile_size:row, col - self.tile_size:col, :] = best
            writer.write(np.uint8(self.out[fnum, ...]))
        writer.release()

def main():
    """
    Generate mosaics.
    """
    mosaic = VideoMosaic()
    vid_extensions = ['mov', 'avi', 'mp4']
    vid_paths = [os.path.join('input', '*.' + ext) for ext in vid_extensions]
    video_files = sorted(sum(map(glob, vid_paths), []))
    for vnum, vid in enumerate(video_files):
        mosaic.generate_mosaic(vid, (800, 1200), 'output/mosaic' + str(vnum) + '.avi')

if __name__ == '__main__':
    main()
