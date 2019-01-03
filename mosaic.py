"""
mosaic.py
Creates video mosaics frame-by-frame.
"""
import argparse
from glob import glob
import os
import sys

import cv2
import numpy as np

def resize_image(image, dims, resize=True):
    """
    Resizes image to supplied dimensions, preserving aspect ratio.
    """
    if resize:
        row_factor = float(dims[0]) / image.shape[0]
        col_factor = float(dims[1]) / image.shape[1]
        factor = np.float(max(row_factor, col_factor))
        image = cv2.resize(image, None, factor, factor, cv2.INTER_AREA)
    resized = image[(image.shape[0] - dims[0])/2:(image.shape[0] - dims[0])/2 + dims[0],
                    (image.shape[1] - dims[1])/2:(image.shape[1] - dims[1])/2 + dims[1]]
    return resized

def boost_color(image, random=True, percent=0.1):
    """
    Boots dominant color channel by specified percentage or
    random channel.
    """
    if random:
        channel = np.random.choice([0, 1, 2])
    else:
        channel = np.argmax(np.sum(image, axis=(0, 1)))
    image[..., channel] = image[..., channel] * (1 + percent)
    image[image > 255] = 255
    return image

class VideoMosaic():
    """
    Generates video mosaics.
    """
    def __init__(self, source_dir, size=20, frame_int=5000, color=None):
        self.frames = None
        self.tiles = None
        self.intensities = None
        self.means = None
        self.tile_size = size
        self.init_source(os.path.join('input', source_dir), frame_int, color)

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

    def init_source(self, source_dir, frame_int, color):
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
                if color:
                    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
                    image = cv2.applyColorMap(np.uint8(image), color)
                image = np.array([resize_image(image, (self.tile_size, self.tile_size))])
                self.tiles = np.concatenate((self.tiles, image), axis=0)
        else:
            vidcap = cv2.VideoCapture(os.path.join(source_dir, 'source.mp4'))
            success, image = vidcap.read()
            count = 0
            while success:
                print('Reading source frame from video ' + str(count))
                # image = image[10:image.shape[0] - 10, ...]
                if np.all(np.std(image, axis=(0, 1)) > 1):
                    cv2.imwrite(os.path.join(source_dir, 'frame{0:04d}.jpg'.format(count/frame_int)), image)
                    image = np.array([resize_image(image, (self.tile_size, self.tile_size))])
                    self.tiles = np.concatenate((self.tiles, image), axis=0)
                count += frame_int
                vidcap.set(cv2.CAP_PROP_POS_MSEC, count)
                success, image = vidcap.read()
        self.intensities = np.average(self.tiles, axis=3, weights=[0.114, 0.587, 0.299])
        self.means = np.average(self.intensities, axis=(1, 2))

    def good_match(self, region, threshold=0):
        """
        Returns best match index for tile region.
        """
        total_diff = np.power(self.means - np.average(region), 2)
        match_values = np.where(total_diff <= np.min(total_diff) * (1 + threshold))
        good = np.random.choice(match_values[0])
        return good

    def generate_mosaic(self, target, dims, out, frate=16):
        """
        Generates and writes video mosaic.
        """
        self.frames = np.zeros((0, dims[0], dims[1], 3))
        self.init_target(target, dims)

        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (dims[1], dims[0]))
        for fnum, frame in enumerate(self.frames):
            print('Generating output frame ' + str(fnum))
            out = np.zeros(frame.shape)
            for row in range(self.tile_size, dims[0] + 1, self.tile_size):
                for col in range(self.tile_size, dims[1] + 1, self.tile_size):
                    region = frame[row - self.tile_size:row, col - self.tile_size:col]
                    best = self.tiles[self.good_match(region), ...]
                    out[row - self.tile_size:row, col - self.tile_size:col, :] = best
            cv2.imshow('frame', np.uint8(out))
            cv2.waitKey(25)
            writer.write(np.uint8(out))
        writer.release()

def main(source, size, color, output=True):
    """
    Generate mosaics.
    """
    mosaic = VideoMosaic(source_dir=source, color=color, size=int(size))
    if output:
        if not os.path.exists(os.path.join('output', source)):
            os.makedirs(os.path.join('output', source))
        vid_extensions = ['mov', 'avi', 'mp4']
        vid_paths = [os.path.join('input', '*.' + ext) for ext in vid_extensions]
        video_files = sorted(sum(map(glob, vid_paths), []))
        for vnum, vid in enumerate(video_files):
            mosaic.generate_mosaic(vid, (800, 1200),
                                   os.path.join('output', source, 'mosaic' + str(vnum) + '.avi'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate video mosaic.')
    parser.add_argument('source', help='Tile source directory.')
    parser.add_argument('size', help='Tile size')
    parser.add_argument('--color', help='Color map to apply to tiles.', default=None)
    args = parser.parse_args()
    main(args.source, args.size, args.color)
