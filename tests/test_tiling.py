from isit.prediction.tiling import TileGenerator

import math
import pytest
import numpy as np


@pytest.mark.parametrize("img_shape,patch_size,border",
                         [((200, 200), (150, 160), (30, 10)),
                          ((200, 200, 1), (150, 160, 1), (30, 10, 0)),
                          ((312, 217, 471, 2), (90, 102, 300, 2),
                           (2, 30, 10, 0))])
def test_tile_generator(img_shape, patch_size, border):

    tile_generator = TileGenerator(img_shape, patch_size, border)

    img = np.random.randint(0, 100, img_shape)
    padd_img = np.pad(img, pad_width=tile_generator.pad_width)

    n_tiles = np.product(
        [math.ceil(s / p) for s, p in zip(img_shape, patch_size)])
    assert n_tiles == len(tile_generator)

    assert img.sum() > 0

    for loc, padded_loc, tile_loc in tile_generator:

        # roundtrip slicing of padded image should give the same tile as from the original image
        img_tile = img[loc]
        padded_img_tile = padd_img[padded_loc][tile_loc]
        np.testing.assert_equal(img_tile, padded_img_tile)

        # size of tile_loc == patch_size except on image border
        for sl, sl_padd, sl_tile, s, p in zip(loc, padded_loc, tile_loc,
                                              img_shape, patch_size):
            if sl.stop < s:
                assert sl_tile.stop - sl_tile.start == p

        img[loc] = 0

    assert img.sum() == 0
