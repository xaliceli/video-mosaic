"""
textures.py
Texture synthesis functions based on Efros & Freeman (2001).
"""
import random

import cv2 
import numpy as np

import boundary as meb

class Texture():
    """
    Synthesizes and transfers textures.

    Attributes:
        dims: Tuple dimensions of desired output.
        w: Size of square region window x window to treat as texture patch.
        k: (w - 1)/2
        adj_wind: Window minus overlap.
        candidates: All possible patches of size w.
        candidates_gray: All possible patches of size w, intensity only.
        filled: Synthesized texture output of dimensions dims.
    """

    def __init__(self, dims, window):
        self.dims = dims
        self.w = int(window)
        self.k = (self.w - 1)/2
        self.adj_wind = self.w - self.w/6
        self.candidates = None
        self.candidates_gray = None

        # Initialize empty array of appropriate size for output, padding for additional rows/cols
        # if desired dimensions are not fully divisible by window size.
        pad_row = (dims[0] / self.adj_wind) * self.adj_wind + self.w - dims[0]
        pad_col = (dims[1] / self.adj_wind) * self.adj_wind + self.w - dims[1]
        self.filled = np.zeros((dims[0] + pad_row, dims[1] + pad_col, dims[2]))

    def generate_candidates(self, source, rotate, correspondence):
        """
        Generates all possible patches in source image.

        Args:
            source (np.array): Image to derive texture from.
            rotate (int): Number of times to rotate patches by for additional candidates.
        """
        # Initialize upper and lower pixel bounds for patch search based on window size
        # and source image dimensions.
        upper_row = source.shape[0] - self.k - 1
        upper_col = source.shape[1] - self.k - 1
        if upper_row == self.k:
            upper_row += 1
        if upper_col == self.k:
            upper_col += 1
        num_unique = (upper_row - self.k) * (upper_col - self.k)

        # Initialize empty array storing pixel values for each possible patch.
        self.candidates = np.zeros((num_unique * (rotate + 1), self.w, self.w, 3))

        # Loop through every possible patch of specified window size.
        for row in range(self.k, upper_row):
            for col in range(self.k, upper_col):
                patch = source[row-self.k:row+self.k+1, col-self.k:col+self.k+1]
                idx = (row - self.k) * (upper_col - self.k) + col - self.k
                self.candidates[idx] = patch

        # Rotate candidates up to 3 times for more options in synthesis.
        if rotate > 0:
            for rotation in range(1, rotate + 1):
                rotated_patches = np.rot90(self.candidates[:num_unique], rotation, (1, 2))
                self.candidates[num_unique * rotation:num_unique * (rotation + 1)] = rotated_patches

        # Calculate grayscale version of each patch.
        if correspondence:
            self.candidates_gray = np.average(self.candidates, axis=3, weights=[0.114, 0.587, 0.299])

    def set_candidates(self, source, rotate, correspondence):
        """
        Reads possible patches from source images

        Args:
            source (np.array): Array of images.
            rotate (int): Number of times to rotate patches by for additional candidates.
        """
        # Initialize empty array storing pixel values for each possible patch.
        self.candidates = source

        # Rotate candidates up to 3 times for more options in synthesis.
        if rotate > 0:
            for rotation in range(1, rotate + 1):
                rotated_patches = np.rot90(self.candidates, rotation, (1, 2))
                self.candidates = np.vstack((self.candidates, rotated_patches))

        # Calculate grayscale version of each patch.
        if correspondence:
            self.candidates_gray = np.average(self.candidates, axis=3, weights=[0.114, 0.587, 0.299])

    def calc_errors(self, up, left):
        """
        Calculate L2 Euclidean distance between each possible patch of specified window size
        and pair of up/left neighbors.

        Args:
            up (np.array): Neighbor patch directly above candidate patch for comparison.
                If None, will only compare with left patch.
            left (np.array): Neighbor patch directly to the left of candidate patch for comparison.
                If None, will only compare with top patch.
        """
        # Intiialize empty array storing errors compared to top and left neighbors.
        errors = np.zeros((self.candidates.shape[0], self.w*2, self.w/6))

        # Loop through every possible patch of specified window size.
        for idx in range(self.candidates.shape[0]):
            patch = self.candidates[idx]

            # Calculates overlap error between current patch and relevant neighbors.
            if up is not None:
                errors[idx, :self.w, :] = meb.overlap_error(np.rot90(up), np.rot90(patch), self.w/6)
            if left is not None:
                errors[idx, self.w:, :] = meb.overlap_error(left, patch, self.w/6)

        return errors

    def good_match(self, correspondence_patch, errors, alpha, min_err=0.05):
        """
        Returns random patch candidate within min_err range of the minimum total distance.

        Args:
            correspondence_patch (np.array): Intensities values for texture transfer.
            errors (np.array): Overlap errors with up and left neighbors.
            alpha (float): Weight of correspondence error compared to overlap error.
            min_err (float): Threshold of acceptable error compared to minimum patch error.

        Returns:
            candidates[match_index] (np.array): Subset of source image selected as
                acceptable match for region based on neighbor values.
            errors[match_index] (np.array): Errors in overlap region with neighbor(s).
        """
        # Calculates total error for each candidate.
        tot_errors = np.sum(errors, axis=(1, 2))

        # If correspondence target is provided, take squared pixel differences
        # and include in total error assessment.
        if correspondence_patch is not None:
            corr_errors = np.power(self.candidates_gray - correspondence_patch, 2)
            tot_errors = alpha * tot_errors + (1 - alpha) * np.sum(corr_errors, axis=(1, 2))

        # Randomly selects a candidate within error tolerance
        match_values = np.where(tot_errors <= np.min(tot_errors) * (1 + min_err))
        match_index = match_values[0][random.randint(0, match_values[0].shape[0] - 1)]

        return self.candidates[match_index], errors[match_index]

    def fill_patch(self, coord, correspondence, alpha):
        """
        Finds acceptable patch and stitches with neighbor(s) along minimum error boundary.

        Args:
            coord (tuple): Row, col coordinates of center of patch in destination image.
            correspondence (np.array): Correspondence target for intensities.
            alpha (float): Weight of correspondence error compared to overlap error.

        Returns:
            stitched (np.array): Image of window size with overlap regions stitched
                along minimum error boundary with neighbor(s).
        """
        # If specified, return neighbor above and to the left of active patch region.
        up_n, left_n = None, None
        if coord[0] > self.k:
            up_n = self.filled[coord[0] - self.adj_wind - self.k:coord[0] -
                               self.adj_wind + self.k + 1,
                               coord[1] - self.k:coord[1] + self.k + 1]
        if coord[1] > self.k:
            left_n = self.filled[coord[0] - self.k:coord[0] + self.k + 1,
                                 coord[1] - self.adj_wind - self.k:coord[1] -
                                 self.adj_wind + self.k + 1]
        if correspondence is not None:
            correspondence_patch = correspondence[coord[0] - self.k:coord[0] + self.k + 1,
                                                  coord[1] - self.k:coord[1] + self.k + 1]
        else:
            correspondence_patch = None

        # Scan source image and find a patch that satisfies error conditions.
        errors = self.calc_errors(up_n, left_n)
        stitched, patch_errors = self.good_match(correspondence_patch, errors, alpha)

        # For each neighbor if specified, calculate minimum error boundary between acceptable
        # patch and neighbor, then stitch together patch with each neighbor along boundary.
        if coord[0] > self.k:
            h_boundary = meb.minimum_error_boundary(patch_errors[:self.w])
            stitched = meb.stitch_images(up_n, stitched,
                                         h_boundary, self.w/6, 'horizontal')
        if coord[1] > self.k:
            v_boundary = meb.minimum_error_boundary(patch_errors[self.w:])
            stitched = meb.stitch_images(left_n, stitched,
                                         v_boundary, self.w/6, 'vertical')

        return stitched

    def fill_image(self, source, correspondence, rotate, alpha):
        """
        Fills in image in raster scan order using texture algorithm from Efros & Freeman (2001).

        Args:
            source (np.array): Image to derive texture from.
            correspondence (np.array): Correspondence target for intensities.
            rotate (int): Number of times to rotate patches by for additional candidates.
            alpha (float): Weight of correspondence error compared to overlap error.
        """
        # Find all available candidate patches and initialize first patch randomly.
        if self.candidates is None:
            self.generate_candidates(source, rotate, correspondence is not None)
            print('Candidates generated.')
        self.filled[:self.w, :self.w] = self.candidates[random.randint(0, self.candidates.shape[0])]

        # Pads correspondence image if available.
        if correspondence is not None:
            pad_row = self.filled.shape[0] - correspondence.shape[0]
            pad_col = self.filled.shape[1] - correspondence.shape[1]
            corr_gray = cv2.cvtColor(np.uint8(correspondence), cv2.COLOR_BGR2GRAY)
            if pad_row > 0 or pad_col > 0:
                correspondence = cv2.copyMakeBorder(corr_gray, 0, pad_row, 0, pad_col,
                                                    cv2.BORDER_REFLECT_101)

        # Fill each patch with an acceptable candidate.
        for row in range(self.k, self.filled.shape[0] + 1, self.adj_wind):
            for col in range(self.k, self.filled.shape[1] + 1, self.adj_wind):
                if row > self.k or col > self.k:
                    stitched = self.fill_patch((row, col), correspondence, alpha)
                    self.filled[row - self.k:row + self.k + 1,
                                col - self.k:col + self.k + 1] = stitched
                    cv2.imshow('filled', np.uint8(self.filled))
                    cv2.waitKey(25)

    def gen_texture(self, source, correspondence=None, rotate=0, alpha=0.5):
        """
        Generates texture from source image and target correspondence, if exists.

        Args:
            source (np.array): Image to derive texture from.
            correspondence (np.array): Correspondence target for intensities.
            rotate (int): Number of times to rotate patches by for additional candidates.
            alpha (float): Weight of correspondence error compared to overlap error.
        """
        self.fill_image(source, correspondence, rotate, alpha)
        return self.filled[:self.dims[0], :self.dims[1]]
