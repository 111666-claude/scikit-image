import numpy as np
import pytest
from numpy.testing import assert_equal

from skimage2.feature import peak_local_max


class TestPeakLocalMax:
    def test_p_norm_default(self):
        image = np.zeros((10, 10))
        image[2, 2] = 1
        image[7, 7] = 1

        # With default (p_norm=2, Euclidean distance), peaks are 7.07 apart
        peaks = peak_local_max(image, min_distance=7)
        assert len(peaks) == 2
        peaks = peak_local_max(image, min_distance=8)
        assert len(peaks) == 1

    def test_exclude_border(self):
        image = np.zeros((5, 5, 5))
        image[[1, 0, 0], [0, 1, 0], [0, 0, 1]] = 1
        image[3, 0, 0] = 1
        image[2, 2, 2] = 1

        # exclude_border=0
        result = peak_local_max(image, min_distance=2, exclude_border=0)
        assert_equal(result, [[0, 0, 1], [2, 2, 2], [3, 0, 0]])

        # Default behavior (exclude_border=1) should be the same
        result = peak_local_max(image, min_distance=2)
        assert_equal(result, [[2, 2, 2]])
        result = peak_local_max(image, min_distance=2, exclude_border=1)
        assert_equal(result, [[2, 2, 2]])

    def test_num_peak_float_error(self):
        image = np.zeros((10, 10))
        peak_local_max(image, num_peaks=1.5)


def test_exact_min_distance_boundary_1d():
    image = np.array([1, 0, 2])

    peaks = peak_local_max(image, min_distance=2, exclude_border=0)

    # Intensity ordering puts the stronger peak first.
    assert_equal(peaks, [[2], [0]])


@pytest.mark.parametrize(
    ("p_norm", "second_peak"),
    [(1, (2, 3)), (2, (3, 4)), (np.inf, (5, 2))],
)
def test_exact_min_distance_boundary_2d(p_norm, second_peak):
    image = np.zeros((8, 8))
    image[0, 0] = 1
    image[second_peak] = 2

    peaks = peak_local_max(image, min_distance=5, p_norm=p_norm, exclude_border=0)

    assert_equal(peaks, [second_peak, [0, 0]])


@pytest.mark.parametrize(
    ("p_norm", "second_peak"),
    [(1, (2, 3)), (2, (3, 4)), (np.inf, (5, 2))],
)
def test_closer_peaks_keep_intensity_and_tie_rules(p_norm, second_peak):
    closer_peak = tuple(coordinate - 1 for coordinate in second_peak)

    # The stronger peak is retained even when it is not first spatially.
    image = np.zeros((8, 8))
    image[0, 0] = 1
    image[closer_peak] = 2
    peaks = peak_local_max(image, min_distance=5, p_norm=p_norm, exclude_border=0)
    assert_equal(peaks, [list(closer_peak)])

    # Equal intensities fall back to the stable/spatial tie rule.
    image = np.zeros((8, 8))
    image[0, 0] = 1
    image[closer_peak] = 1
    peaks = peak_local_max(image, min_distance=5, p_norm=p_norm, exclude_border=0)
    assert_equal(peaks, [[0, 0]])


def test_explicit_footprint_keeps_its_own_convention():
    image = np.array([1, 0, 2])

    # This inclusive footprint compares points exactly two indices apart;
    # users opting into it retain the previous maximum-filter behavior.
    peaks = peak_local_max(
        image,
        min_distance=2,
        exclude_border=0,
        footprint=np.ones(5, dtype=bool),
    )

    assert_equal(peaks, [[2]])


def test_exact_min_distance_with_labels():
    image = np.array([1, 0, 2])
    labels = np.ones_like(image)

    peaks = peak_local_max(image, min_distance=2, exclude_border=0, labels=labels)

    assert_equal(peaks, [[2], [0]])
