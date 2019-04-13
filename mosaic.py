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

from texture import Texture

def resize_image(image, dims, resize=True):
    """
    Resizes image to supplied dimensions, preserving aspect ratio.
    """
    if resize:
        row_factor = float(dims[0]) / image.shape[0]
        col_factor = float(dims[1]) / image.shape[1]
        factor = np.float(max(row_factor, col_factor))
        if factor < 1:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    resized = image[int((image.shape[0] - dims[0])/2):int((image.shape[0] - dims[0])/2 + dims[0]),
                    int((image.shape[1] - dims[1])/2):int((image.shape[1] - dims[1])/2 + dims[1])]
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

    def generate_mosaic(self, target, dims, blur, out, vtype, frate=16):
        """
        Generates and writes video mosaic.
        """
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (dims[1], dims[0]))
        reader = cv2.VideoCapture(target)
        success, image = reader.read()
        count = 0
        texturize = None
        while success:
            print('Reading target frame ' + str(count))
            frame = resize_image(image, dims)
            print('Generating output frame ' + str(count))
            if vtype == 'mosaic':
                out_frame = np.zeros(frame.shape)
                for row in range(self.tile_size, dims[0] + 1, self.tile_size):
                    for col in range(self.tile_size, dims[1] + 1, self.tile_size):
                        region = frame[row - self.tile_size:row, col - self.tile_size:col]
                        best = self.tiles[self.good_match(region), ...]
                        out_frame[row - self.tile_size:row, col - self.tile_size:col, :] = best
            elif vtype == 'texture':
                if texturize is None:
                    texturize = Texture((dims[0], dims[1], 3), 25)
                    texturize.set_candidates(self.tiles, 3, True)
                out_frame = texturize.gen_texture(None, frame, 3, .5)
            if blur:
                out_frame = cv2.GaussianBlur(out_frame, (21, 21), 0)
            writer.write(np.uint8(out_frame))
            success, image = reader.read()
            count += 1
        reader.release()
        writer.release()

def main(source, target, size, out_size, color, blur, vtype='mosaic', output=True):
    """
    Generate mosaics.
    """
    mosaic = VideoMosaic(source_dir=source, color=color, size=int(size))
    if output:
        if not os.path.exists(os.path.join('output', source)):
            os.makedirs(os.path.join('output', source))
        vid_extensions = ['mov', 'avi', 'mp4']
        vid_paths = [os.path.join('input', target, '*.' + ext) for ext in vid_extensions]
        video_files = sorted(sum(map(glob, vid_paths), []))
        for vnum, vid in enumerate(video_files):
            mosaic.generate_mosaic(vid, out_size, blur,
                                   os.path.join('output', source, 'mosaic' + str(vnum) + '.avi'),
                                   vtype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate video mosaic.')
    parser.add_argument('source', help='Tile source directory.')
    parser.add_argument('target', help='Target source directory.')
    parser.add_argument('size', help='Tile size.')
    parser.add_argument('vtype', help='Mosaic or texture.')
    parser.add_argument('--out_size', help='Size of output video as tuple in form (rows, columns).',
                        default=(1600, 2400))
    parser.add_argument('--color', help='OpenCV color map to apply to tiles.', default=None)
    parser.add_argument('--blur', help='Apply a Gaussian blur to frames.', default=False)
    args = parser.parse_args()
    main(args.source, args.target, args.size, args.out_size, args.color, args.blur, args.vtype)
