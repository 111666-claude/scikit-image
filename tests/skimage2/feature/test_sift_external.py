"""Tests for SIFT descriptor extraction at externally supplied keypoints."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from _skimage2 import data
from _skimage2.feature import SIFT
from _skimage2.util.dtype import _convert

img = data.coins()


@pytest.fixture(scope="module")
def detected():
    sift = SIFT()
    sift.detect_and_extract(img)
    return sift


def test_extract_external_matches_detection_prefix(detected):
    # Requirement: run the detection path on a fixed image, then re-extract
    # descriptors through the external entry and compare item by item.
    keypoints = detected.positions[:10]
    sigmas = detected.sigmas[:10]

    extractor = SIFT()
    descriptors = extractor.extract(img, keypoints, sigmas)

    assert descriptors.shape == (10, 128)
    assert descriptors.dtype == np.uint8
    assert_array_equal(descriptors, detected.descriptors[:10])
    # state follows input order and is exposed like the internal path
    assert_array_equal(extractor.descriptors, descriptors)
    assert extractor.positions.shape == (10, 2)
    assert extractor.keypoints.shape == (10, 2)


def test_extract_descriptors_alias_and_custom_layout(detected):
    # The hard-coded descriptor fixture uses a 2x2 x 4 orientation layout;
    # the same points must reproduce it through the named entry point.
    sift = SIFT(n_hist=2, n_ori=4)
    sift.detect_and_extract(img)

    other = SIFT(n_hist=2, n_ori=4)
    descriptors = other.extract_descriptors(
        img, sift.positions[:10], sift.sigmas[:10]
    )
    assert descriptors.shape == (10, 16)
    assert_array_equal(descriptors, sift.descriptors[:10])


def test_extract_external_full_internal_roundtrip(detected):
    # Supplying the internal octave/scale/orientation mapping reproduces
    # every descriptor exactly, across all octaves.
    other = SIFT()
    descriptors = other.extract(
        img,
        detected.positions,
        detected.sigmas,
        detected.scales,
        detected.octaves,
        detected.orientations,
    )
    assert_array_equal(descriptors, detected.descriptors)


def test_extract_external_preserves_input_order(detected):
    rng = np.random.default_rng(123)
    permutation = rng.permutation(len(detected.positions))

    other = SIFT()
    descriptors = other.extract(
        img,
        detected.positions[permutation],
        detected.sigmas[permutation],
        detected.scales[permutation],
        detected.octaves[permutation],
        detected.orientations[permutation],
    )
    assert_array_equal(descriptors, detected.descriptors[permutation])
    assert_array_equal(other.octaves, detected.octaves[permutation])


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8"])
def test_extract_external_dtypes(detected, dtype):
    image = _convert(img, dtype)
    descriptors = SIFT().extract(
        image, detected.positions[:5], detected.sigmas[:5]
    )
    assert descriptors.shape == (5, 128)
    assert descriptors.dtype == np.uint8


@pytest.mark.parametrize(
    "empty_keypoints",
    [
        np.empty((0, 2)),
        np.array([]),
        [],
        np.zeros((0, 3)),
    ],
)
def test_extract_external_empty(empty_keypoints):
    sift = SIFT()
    descriptors = sift.extract(img, empty_keypoints)
    assert descriptors.shape == (0, 128)
    assert descriptors.dtype == np.uint8
    assert sift.descriptors.shape == (0, 128)
    assert sift.positions.shape == (0, 2)

    scales = np.empty(0)
    custom = SIFT(n_hist=2, n_ori=4).extract_descriptors(
        img, empty_keypoints, scales
    )
    assert custom.shape == (0, 16)


def test_extract_external_empty_does_not_detect():
    # A textureless image makes the internal detection raise; empty external
    # input must return an empty result instead of re-detecting.
    flat = np.ones((128, 128))
    descriptors = SIFT().extract(
        flat, np.empty((0, 2)), np.empty((0,))
    )
    assert descriptors.shape == (0, 128)


@pytest.mark.parametrize(
    "positions,sigmas",
    [
        (np.zeros((4, 3)), np.ones(4)),       # wrong second dimension
        (np.zeros(2), np.ones(1)),            # 1D instead of (N, 2)
        (np.zeros((4, 2)), np.ones(3)),       # scale length mismatch
        (np.zeros((4, 2)), np.ones((4, 1))),  # scales must be 1D
    ],
)
def test_extract_external_bad_shape(positions, sigmas):
    with pytest.raises(ValueError):
        SIFT().extract(img, positions, sigmas)


@pytest.mark.parametrize(
    "position",
    [
        [-1.0, 10.0],
        [10.0, -1.0],
        [float(img.shape[0]), 10.0],
        [10.0, float(img.shape[1])],
        [1e6, 1e6],
    ],
)
def test_extract_external_out_of_bounds(position):
    with pytest.raises(ValueError):
        SIFT().extract(img, np.array([position]), np.array([1.6]))


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_extract_external_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        SIFT().extract(
            img, np.array([[50.0, 50.0]]), np.array([sigma])
        )


def test_extract_external_nan_position():
    with pytest.raises(ValueError):
        SIFT().extract(
            img, np.array([[np.nan, 50.0]]), np.array([1.6])
        )


def test_extract_external_corner_point():
    # In-image border points are extracted (with clipped patches), not
    # dropped, and keep the one-row-per-input contract.
    descriptors = SIFT().extract(
        img, np.array([[0.0, 0.0]]), np.array([1.6])
    )
    assert descriptors.shape == (1, 128)


def test_extract_external_flat_patch_is_zero_descriptor():
    flat = np.ones((128, 128))
    descriptors = SIFT().extract(
        flat, np.array([[64.0, 64.0]]), np.array([1.6])
    )
    assert_equal(descriptors, np.zeros((1, 128), dtype=np.uint8))


def test_legacy_extract_without_arguments_unchanged():
    detector = SIFT()
    detector.detect(img)
    detector.extract(img)

    both = SIFT()
    both.detect_and_extract(img)

    assert_array_equal(detector.descriptors, both.descriptors)
