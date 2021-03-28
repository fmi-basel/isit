import itertools


class TileGenerator():
    '''Generate slices need to process an image by tiles'''
    def __init__(self, img_shape, patch_size, border):
        '''
        Args:
            img_shape: image shape
            patch_size: size of the patch, excluding border
            border: margin around the patch
            
            (use the pad_with to padd the image)
        '''

        corners_coords = tuple(
            list(range(0, s, p))
            for s, b, p in zip(img_shape, border, patch_size))
        corners = list(itertools.product(*corners_coords))

        locs = []
        padded_locs = []
        tile_locs = []
        for corner in corners:
            loc = tuple(
                slice(c, min(s, c + p))
                for c, p, s in zip(corner, patch_size, img_shape))
            tile_loc = tuple(
                slice(b, sl.stop - sl.start + b) for sl, b in zip(loc, border))
            padded_loc = tuple(
                slice(c, c + p + 2 * b)
                for c, p, b in zip(corner, patch_size, border))

            locs.append(loc)
            tile_locs.append(tile_loc)
            padded_locs.append(padded_loc)

        self.locs = locs
        self.tile_locs = tile_locs
        self.padded_locs = padded_locs

        self.pad_width = tuple(
            (b, p - max(1, s % p) + b)
            for s, b, p in zip(img_shape, border, patch_size))

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        '''
        
        Return:
            loc: slices to get a tile from the original images
            padded_loc: slices to get the corresponding tile on the padded image
            tile_loc: slices in tile coordinates to remove the border
        '''

        return self.locs[idx], self.padded_locs[idx], self.tile_locs[idx]
