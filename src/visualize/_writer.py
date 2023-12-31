"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
# from __future__ import annotations

# from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

# if TYPE_CHECKING:
#     DataType = Union[Any, Sequence[Any]]
#     FullLayerData = Tuple[DataType, dict, str]

import os
from shutil import make_archive, rmtree
from typing import Any

from napari.qt import thread_worker
from tifffile import imsave


def write_single_image(path: str, data: Any, meta: dict):
    """Writes a single image layer"""
    if str(path).endswith(".zip"):
        path = str(path)[:-4]
    if not data.ndim == 3:
        return None

    @thread_worker()
    def write_tiffs(data, dir_pth):
        for i, t_slice in enumerate(data):
            tiff_nme = f"seg{str(i).zfill(3)}.tif"
            tiff_pth = os.path.join(dir_pth, tiff_nme)
            imsave(tiff_pth, t_slice)

    def zip_dir():
        make_archive(path, "zip", path)
        rmtree(path)

    os.mkdir(path)
    layer_dir_pth = os.path.join(path, f"01_AUTO/SEG")
    os.makedirs(layer_dir_pth)
    if len(data.shape) == 2:
        data = [data]
    worker = write_tiffs(data, layer_dir_pth)
    worker.start()
    worker.finished.connect(zip_dir)
    return path + ".zip"


# def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
#     """Writes multiple layers of different types."""

#     # implement your writer logic here ...

#     # return path to any file(s) that were successfully written
#     return [path]
