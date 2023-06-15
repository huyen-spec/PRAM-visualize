"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
# import numpy as np


# def napari_get_reader(path):
#     """A basic implementation of a Reader contribution.

#     Parameters
#     ----------
#     path : str or list of str
#         Path to file, or list of paths.

#     Returns
#     -------
#     function or None
#         If the path is a recognized format, return a function that accepts the
#         same path or list of paths, and returns a list of layer data tuples.
#     """
#     if isinstance(path, list):
#         # reader plugins may be handed single path, or a list of paths.
#         # if it is a list, it is assumed to be an image stack...
#         # so we are only going to look at the first file.
#         path = path[0]

#     # if we know we cannot read the file, we immediately return None.
#     if not path.endswith(".npy"):
#         return None

#     # otherwise we return the *function* that can read ``path``.
#     return reader_function


# def reader_function(path):
#     """Take a path or list of paths and return a list of LayerData tuples.

#     Readers are expected to return data as a list of tuples, where each tuple
#     is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
#     both optional.

#     Parameters
#     ----------
#     path : str or list of str
#         Path to file, or list of paths.

#     Returns
#     -------
#     layer_data : list of tuples
#         A list of LayerData tuples where each tuple in the list contains
#         (data, metadata, layer_type), where data is a numpy array, metadata is
#         a dict of keyword arguments for the corresponding viewer.add_* method
#         in napari, and layer_type is a lower-case string naming the type of
#         layer. Both "meta", and "layer_type" are optional. napari will
#         default to layer_type=="image" if not provided
#     """
#     # handle both a string and a list of strings
#     paths = [path] if isinstance(path, str) else path
#     # load all files into array
#     arrays = [np.load(_path) for _path in paths]
#     # stack arrays into single array
#     data = np.squeeze(np.stack(arrays))

#     # optional kwargs for the corresponding viewer.add_* method
#     add_kwargs = {}

#     layer_type = "image"  # optional, default is "image"
#     return [(data, add_kwargs, layer_type)]




import glob
import os
import re
import warnings
from pathlib import Path

import dask
import dask.array as da
import tifffile
from napari.utils import progress

SEQ_REGEX = r"(.*)/([0-9]{2,})$"
GT_REGEX = r"(.*)/([0-9]{2,})(_(?:GT|AUTO))/SEG$"

SEQ_TIF_REGEX = rf'{SEQ_REGEX[:-1]}/t([0-9]{{3}}){"."}tif$'
GT_TIF_REGEX = rf'{GT_REGEX[:-1]}/(?:man_)?seg([0-9]{{3}}){"."}tif$'


def napari_get_reader(path):
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    is_gt = re.match(GT_REGEX, path)
    if not is_gt:
        return None
    all_tifs = glob.glob(path + "/*.tif")
    print(path + "/*.tif")
    if not all_tifs:
        return None
    is_gt_tifs = all([re.match(GT_TIF_REGEX, pth) for pth in all_tifs])
    if not is_gt_tifs:
        return None

    return reader_function


def reader_function(path):
    path = os.path.normpath(path)
    
    print("path", path)
    gt_match = re.match(GT_REGEX, path)

    parent_dir_pth = Path(path).parent.parent.absolute()
    seq_number = gt_match.groups()[1]
    sister_sequence_pth = os.path.join(parent_dir_pth, seq_number)

    n_frames = None
    if not os.path.exists(sister_sequence_pth):
        warnings.warn(
            f"Can't find image for ground truth at {path}. Reading without knowing number of frames..."
        )
    else:
        latest_tif_pth = sorted(glob.glob(sister_sequence_pth + "/*.tif"))[-1]
        n_frames = (
            int(re.match(SEQ_TIF_REGEX, latest_tif_pth).groups()[-1]) + 1
        )

    all_tifs = sorted(
        pth
        for pth in glob.glob(path + "/*.tif")
        if re.match(GT_TIF_REGEX, pth)
    )
    tif_shape = None
    tif_dtype = None
    with tifffile.TiffFile(all_tifs[0]) as im_tif:
        tif_shape = im_tif.pages[0].shape
        tif_dtype = im_tif.pages[0].dtype
    if not n_frames:
        n_frames = len(all_tifs)
    im_stack = [
        da.zeros(shape=tif_shape, dtype=tif_dtype) for _ in range(n_frames)
    ]

    @dask.delayed
    def read_im(tif_pth):
        with tifffile.TiffFile(tif_pth) as im_tif:
            im = im_tif.pages[0].asarray()
        return im

    for tif_pth in progress(all_tifs):
        frame_index = int(re.match(GT_TIF_REGEX, tif_pth).groups()[-1])
        im = da.from_delayed(
            read_im(tif_pth), shape=tif_shape, dtype=tif_dtype
        )
        im_stack[frame_index] = im
    layer_data = da.stack(im_stack)

    layer_type = "labels"
    layer_name = f"{gt_match.group(2)}{gt_match.group(3)}"
    add_kwargs = {"name": layer_name}

    return [(layer_data, add_kwargs, layer_type)]