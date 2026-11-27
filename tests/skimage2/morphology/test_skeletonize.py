import numpy as np
import pytest
from numpy.testing import assert_array_equal
import scipy.ndimage as ndi

from _skimage2 import io, draw
from _skimage2._shared.testing import fetch
from _skimage2.data import binary_blobs
from _skimage2.morphology import medial_axis, skeletonize, thin
from _skimage2.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts


class TestSkeletonize:
    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_no_foreground(self, method):
        image = np.zeros((5, 5))
        result = skeletonize(image, method=method)
        assert_array_equal(result, np.zeros((5, 5)))

    @pytest.mark.parametrize(
        "ndim,method", [(1, "zhang"), (3, "zhang"), (1, "lee"), (4, "lee")]
    )
    def test_wrong_ndim(self, ndim, method):
        image = np.zeros((5,) * ndim, dtype=bool)
        with pytest.raises(ValueError):
            skeletonize(image, method=method)

    def test_wrong_method(self):
        image = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError):
            skeletonize(image, method="foo")

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_skeletonize_all_foreground(self, method):
        image = np.ones((3, 4), dtype=bool)
        result = skeletonize(image, method=method)
        if method == "zhang":
            expected = np.array([[0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
        else:  # "lee"
            expected = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_single_point(self, method):
        image = np.zeros((5, 5), dtype=bool)
        image[3, 3] = 1
        result = skeletonize(image, method=method)
        assert_array_equal(result, image)

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_vec_1d(self, method):
        # Corner case of a 2D image, which is a 1D vector
        image = np.ones((5, 1), dtype=bool)
        result = skeletonize(image, method=method)
        assert_array_equal(result, image)

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_already_thinned(self, method):
        image = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = skeletonize(image, method=method)
        assert_array_equal(result, image)

    def test_output(self):
        image = io.imread(fetch("data/bw_text.png"), as_gray=True)

        # make black the foreground
        image = image == 0
        result = skeletonize(image)

        expected = np.load(fetch("data/bw_text_skeleton.npy"))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_num_neighbors(self, method, dtype):
        # an empty image
        image = np.zeros((300, 300), dtype=dtype)

        # foreground object 1
        image[10:-10, 10:100] = 1
        image[-100:-10, 10:-10] = 2
        image[10:-10, -100:-10] = 3

        # foreground object 2
        rs, cs = draw.line(250, 150, 10, 280)
        for i in range(10):
            image[rs + i, cs] = 4
        rs, cs = draw.line(10, 150, 250, 280)
        for i in range(20):
            image[rs + i, cs] = 5

        # foreground object 3
        ir, ic = np.indices(image.shape)
        circle1 = (ic - 135) ** 2 + (ir - 150) ** 2 < 30**2
        circle2 = (ic - 135) ** 2 + (ir - 150) ** 2 < 20**2
        image[circle1] = 1
        image[circle2] = 0
        result = skeletonize(image, method=method).astype(np.uint8)

        # there should never be a 2x2 block of foreground pixels in a skeleton
        mask = np.array([[1, 1], [1, 1]], np.uint8)
        blocks = ndi.correlate(result, mask, mode="constant")
        assert not np.any(blocks == 4)

    def test_lut_fix(self):
        image = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = skeletonize(image)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        assert np.all(result == expected)

    @pytest.mark.parametrize("ndim,method", [(2, "zhang"), (2, "lee"), (3, "lee")])
    @pytest.mark.parametrize("dtype", [bool, np.uint8])
    def test_input_not_modified(self, method, ndim, dtype):
        # Skeletonize must not modify the input image
        image = np.ones((3,) * ndim, dtype=dtype)
        image = np.pad(image, 1)
        original = image.copy()
        _ = skeletonize(image, method=method)
        np.testing.assert_array_equal(image, original)

    @pytest.mark.parametrize("method", ["zhang", "lee"])
    def test_input_float_conv(self, method):
        # Check that the floats are correctly handled. Also check non-contiguous input
        rng = np.random.RandomState(3127858794)
        image = rng.random((16, 16))[::2, ::2]
        image[image < 0.5] = 0.0

        original = image.copy()
        result = skeletonize(image, method=method)

        assert result.dtype == bool
        assert_array_equal(image, original)

    def test_two_hole_image_vs_fiji(self):
        # Test a simple 2D image against FIJI
        image = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = skeletonize(image, method="lee")
        assert_array_equal(result, expected)

    def test_3d_vs_fiji(self):
        # Generate an image with blobs and compare its skeleton
        # to the one generated by FIJI (Plugins>Skeleton->Skeletonize)
        image = binary_blobs(
            shape=(32, 32, 32),
            blob_size=0.05 * 32,
            boundary_mode='nearest',
            rng=1234,
        )
        image = image[:-2, ...]

        result = skeletonize(image)
        expected = io.imread(fetch("data/_blobs_3d_fiji_skeleton.tif")).astype(bool)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (16, 16, 16),
            (16, 16, 15),
            (15, 15, 15),
            (8, 8, 8),
            (6, 6, 6),
            (4, 4, 4),
            (2, 2, 2),
            (16, 16, 14),
            (4, 4, 5),
            (2, 2, 3),
        ],
    )
    def test_solid_box_does_not_vanish(self, shape):
        # Filled boxes with even-sized extents used to be eroded away
        # entirely, while odd-sized ones were not. The skeleton of a solid,
        # simply connected object must stay non-empty and connected,
        # regardless of size parity.
        image = np.ones(shape, dtype=bool)
        result = skeletonize(image)
        structure = ndi.generate_binary_structure(3, 3)
        assert result.any()
        assert ndi.label(result, structure=structure)[1] == 1
        # the skeleton stays within the original foreground
        assert np.all(result <= image)

    def test_even_cube_skeleton(self):
        # Exact regression test: a filled 16^3 cube is reduced to a single
        # central voxel instead of vanishing.
        image = np.ones((16, 16, 16), dtype=bool)
        result = skeletonize(image)
        expected = np.zeros_like(image)
        expected[8, 8, 8] = True
        assert_array_equal(result, expected)

    def test_solid_object_axis_permutations_and_flips(self):
        # An asymmetric, simply connected solid: thinning must not disappear
        # or gain thickness depending on accidental axis order or flips.
        image = np.zeros((16, 16, 16), dtype=bool)
        image[2:14, 3:13, 4:12] = True
        image[10:14, 6:10, 1:4] = True  # protruding arm

        structure = ndi.generate_binary_structure(3, 3)
        block2x2x2 = np.ones((2, 2, 2), dtype=np.uint8)

        variants = [image]
        variants += [
            np.rot90(image, k, axes)
            for axes in [(0, 1), (0, 2), (1, 2)]
            for k in (1, 2, 3)
        ]
        variants += [np.flip(image, axis) for axis in (0, 1, 2)]

        for variant in variants:
            result = skeletonize(variant)
            # non-empty, single connected component, within the foreground
            assert result.any()
            assert ndi.label(result, structure=structure)[1] == 1
            assert np.all(result <= variant)
            # no 2x2x2 block of foreground: the skeleton stays thin
            thick = ndi.correlate(
                result.astype(np.uint8), block2x2x2, mode="constant"
            )
            assert thick.max() < 8

    def test_skeletonize_3d_preserves_components(self):
        # Every connected component of the input must survive thinning,
        # including solid even-sized ones, and holes must not be filled.
        structure = ndi.generate_binary_structure(3, 3)

        # two separate filled even-sized cubes
        image = np.zeros((16, 16, 16), dtype=bool)
        image[1:5, 1:5, 1:5] = True
        image[10:14, 10:14, 10:14] = True
        result = skeletonize(image)
        assert ndi.label(result, structure=structure)[1] == 2

        # a cube with a straight tunnel through it (genus 1)
        image = np.ones((16, 16, 16), dtype=bool)
        image[6:10, 6:10, :] = False
        result = skeletonize(image)
        assert result.any()
        assert ndi.label(result, structure=structure)[1] == 1


class TestThin:
    @property
    def input_image(self):
        # Image to test thinning with
        ii = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 3, 4, 5, 0],
                [0, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 6, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        return ii

    def test_all_zeros(self):
        image = np.zeros((10, 10), dtype=bool)
        assert np.all(thin(image) == False)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_thin_copies_input(self, dtype):
        """Ensure thinning does not modify the input image."""
        image = self.input_image.astype(dtype)
        original = image.copy()
        thin(image)
        np.testing.assert_array_equal(image, original)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_iter_1(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image, 1).astype(bool)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_noiter(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image).astype(bool)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        assert_array_equal(result, expected)

    def test_baddim(self):
        for ii in [np.zeros(3, dtype=bool), np.zeros((3, 3, 3), dtype=bool)]:
            with pytest.raises(ValueError):
                thin(ii)

    def test_lut_generation(self):
        g123, g123p = _generate_thin_luts()

        assert_array_equal(g123, G123_LUT)
        assert_array_equal(g123p, G123P_LUT)


class TestMedialAxis:
    def test_all_zeros(self):
        result = medial_axis(np.zeros((10, 10), dtype=bool))
        assert np.all(result == False)

    def test_all_zeros_masked(self):
        result = medial_axis(
            np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)
        )
        assert np.all(result == False)

    @pytest.mark.parametrize("dtype", [bool, float, int])
    def test_vertical_line(self, dtype):
        # Image is a thick vertical line (see gh-3861)
        image = np.zeros((9, 9), dtype=dtype)
        image[:, 2] = 1
        image[:, 3] = 2
        image[:, 4] = 3

        expected = np.full(image.shape, False)
        expected[:, 3] = True

        result = medial_axis(image)
        assert_array_equal(result, expected)

    def test_rectangle(self):
        image = np.zeros((9, 15), dtype=bool)
        image[1:-1, 1:-1] = True
        # Excepted are four diagonals from the corners, meeting in a horizontal line
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = medial_axis(image)
        assert np.all(result == expected)
        result, distance = medial_axis(image, return_distance=True)
        assert distance.max() == 4

    def test_rectange_with_hole(self):
        image = np.zeros((9, 15), dtype=bool)
        image[1:-1, 1:-1] = True
        image[4, 4:-4] = False
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        result = medial_axis(image)
        assert np.all(result == expected)

    def test_narrow_image(self):
        # Image is a 1-pixel thin strip
        image = np.zeros((1, 5), dtype=bool)
        image[:, 1:-1] = True
        result = medial_axis(image)
        assert np.all(result == image)
