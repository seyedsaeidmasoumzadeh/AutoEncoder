import os, glob
import numpy as np
import scipy.misc


class ImageUtils(object):


    """
      read raw data from a given directory, resize them and save them to the another directory
    """
    def save_raw2resized(self, raw_dir=None, processed_dir=None, img_shape=None):

        (ypixels_force, xpixels_force) = img_shape
        gray_scale = False

        raw_filenames_list, n_files = self._extract_filenames(raw_dir, 1)

        for i, raw_filename in enumerate(raw_filenames_list):

            img_raw = self._read_img(raw_filename, gray_scale=gray_scale)
            img_resized = self._force_resize_img(img_raw, ypixels_force, xpixels_force)
            name, tag = self._extract_name_tag(raw_filename)
            processed_shortname = name + "_resized." + tag
            processed_filename = os.path.join(processed_dir, processed_shortname)
            self._save_img(img_resized, processed_filename)

            # Print process progress
            print("[{0}/{1}] Resized and saved to '{2}'...".format(
                i+1, n_files, processed_filename))

    """
     load images from a given directory to a numpy array
    """
    def load_raw2resizednorm(self, img_dir=None, img_shape=None):

        (ypixels_force, xpixels_force) = img_shape
        gray_scale = False

        # Extract filenames from dir
        raw_filenames_list, n_files = self._extract_filenames(img_dir, 1)

        img_list = []
        for i, raw_filename in enumerate(raw_filenames_list):

            # Read raw image
            img_raw = self._read_img(raw_filename, gray_scale=gray_scale)

            # Resize image (if not of shape (ypixels_force, xpixels_force))
            img_resized = img_raw
            if img_raw.shape[:2] != img_shape:
                img_resized = self._force_resize_img(img_resized, ypixels_force, xpixels_force)

            # Normalize image
            img_resizednorm = self._normalize_img_data(img_resized)

            # Append processed image to image list
            img_list.append(img_resizednorm)

            # Print process progress
            print("[{0}/{1}] Loaded and processed '{2}'...".format(
                i + 1, n_files, raw_filename))

        # Convert image list to numpy array
        img_list = np.array(img_list)

        # Make tests
        if img_list.shape[0] != n_files:
            raise Exception("Inconsistent number of loading images!")
        if img_list.shape[1] != ypixels_force:
            raise Exception("Inconsistent ypixels loading images!")
        if img_list.shape[2] != xpixels_force:
            raise Exception("Inconsistent xpixels loading images!")
        if img_list.shape[3] != 3:
            raise Exception("Inconsistent RGB loading images!")

        return img_list, raw_filenames_list

    """
         load an image to a numpy array
    """

    def load_img_resizednorm(self, img_path=None, img_shape=None):

        (ypixels_force, xpixels_force) = img_shape
        gray_scale = False
        img_raw = self._read_img(img_path, gray_scale=gray_scale)
        img_resized = img_raw
        if img_raw.shape[:2] != img_shape:
            img_resized = self._force_resize_img(img_resized, ypixels_force, xpixels_force)

        return self._normalize_img_data(np.array([img_resized]))

    """
           Flatten image data
    """

    def flatten_img_data(self, x_data):
        n_data = x_data.shape[0]
        flatten_dim = np.prod(x_data.shape[1:])
        x_data_flatten = x_data.reshape((n_data, flatten_dim))
        return x_data_flatten

    def _read_img(self, img_filename, gray_scale=False):
        if gray_scale:
            img = np.array(scipy.misc.imread(img_filename, flatten=gray_scale))
        else:
            img = np.array(scipy.misc.imread(img_filename, mode='RGB'))
        return img

    def _save_img(self, img, save_filename):
        scipy.misc.imsave(save_filename, img)
        return

    def _normalize_img_data(self, x_data):
        x_data_norm = x_data.astype('float32') / 255.  # normalize values [0,1]
        return x_data_norm

    def _force_resize_img(self, img, ypixels_force, xpixels_force):
        img_resized = np.array(scipy.misc.imresize(img, (ypixels_force, xpixels_force)))
        return img_resized

    def _extract_name_tag(self, full_filename):
        shortname = full_filename[full_filename.rfind("/") + 1:]  # filename (within its last directory)
        shortname_split = shortname.split('.')
        name = ''
        n_splits = len(shortname_split)
        for i, x in enumerate(shortname_split):
            if i == 0:
                name = name + x
            elif i == n_splits - 1:
                break
            else:
                name = name + '.' + x
        tag = shortname_split[-1]
        return name, tag


    def _extract_filenames(self, dir, frac_take):
        filenames_list = glob.glob(dir + "/*")
        n_files = len(filenames_list)
        if n_files == 0:
            raise Exception("There are no files in {0}!".format(dir))
        if frac_take != 1:
            n_files = int(frac_take * n_files)
            filenames_list = filenames_list[:n_files]
        return filenames_list, n_files


