import numpy as np
from typing import Tuple
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.patches as patches

X_SIZE, Y_SIZE, Z_SIZE = 16, 8, 8


def _retrieve_slice(image, plane, loc):
    if plane == "Sagittal":
        image_slice = image[loc, :, :]
    elif plane == "Coronal":
        image_slice = image[:, loc, :]
    elif plane == "Axial":
        image_slice = image[:, :, loc]
    else:
        raise ValueError(f"Unexpected view plane value")

    # image_slice = np.rot90(image_slice, -1)
    # image_slice = np.fliplr(image_slice)

    return image_slice


def render(image: np.ndarray):
    if len(image.shape) == 4:
        assert image.shape[0] == 1, f"Can only render one image at a time, got {image.shape[0]} images instead"
        image = image[0]
    shape = image.shape
    assert shape[0] == shape[1] and shape[1] == shape[2]
    edge_length = shape[0]

    def _view_func(plane="Axial", loc=0):
        image_slice = _retrieve_slice(image, plane, loc)
        plt.imshow(image_slice, cmap='gray', origin='lower')
        plt.show()

    widgets.interact(_view_func, plane=["Axial", "Coronal", "Sagittal"], loc=(0, 127))


def render_highlighted_chunk(image: np.ndarray, loc: Tuple[int, int, int]):
    if len(image.shape) == 4:
        assert image.shape[0] == 1, f"Can only render one image at a time, got {image.shape[0]} images instead"
        image = image[0]
    shape = image.shape
    assert shape[0] == shape[1] and shape[1] == shape[2]
    edge_length = shape[0]

    x, y, z = loc
    assert 0 <= x < X_SIZE, f"x must be within range [0, {X_SIZE})"
    assert 0 <= y < Y_SIZE, f"x must be within range [0, {Y_SIZE})"
    assert 0 <= z < Z_SIZE, f"x must be within range [0, {Z_SIZE})"

    im_x_step, im_y_step, im_z_step = 128 / X_SIZE, 128 / Y_SIZE, 128 / Z_SIZE
    im_x, im_y, im_z = x * im_x_step, y * im_y_step, z * im_z_step

    def _view_func(plane="Axial", loc=0):
        image_slice = _retrieve_slice(image, plane, loc)
        fig, ax = plt.subplots()
        ax.imshow(image_slice, cmap='gray', origin='lower')

        if plane == "Sagittal" and im_x <= loc < im_x + im_x_step:
            has_rect = True
            rect_pos = (im_y, im_z)
            rect_size = (im_y_step, im_z_step)

        elif plane == "Coronal" and im_y <= loc < im_y + im_y_step:
            has_rect = True
            rect_pos = (im_x, im_z)
            rect_size = (im_x_step, im_z_step)
        elif plane == "Axial" and im_z <= loc < im_z + im_z_step:
            has_rect = True
            rect_pos = (im_x, im_y)
            rect_size = (im_x_step, im_y_step)
        else:
            has_rect = False
            rect_pos, rect_size = None, None

        if has_rect:
            rect = patches.Rectangle(rect_pos, *rect_size, linewidth=1, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
        fig.text(.5, 0, f"Chunk ({x}, {y}, {z}), status: {'visible' if has_rect else 'not visible'}")

        plt.show()

    widgets.interact(_view_func, plane=["Axial", "Coronal", "Sagittal"], loc=(0, 127))
