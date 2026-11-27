import numpy as np
from _skimage2._shared.testing import assert_array_equal, assert_allclose

from _skimage2.draw import ellipsoid, ellipsoid_stats, rectangle
from _skimage2._shared import testing


def _analytic_ellipsoid(a, b, c, rotation, spacing):
    """Independent reference implementation using the ellipsoid equation.

    Evaluates the signed quadratic form of the rotated ellipsoid on the same
    grid that `ellipsoid` uses, without reusing any of its internals.
    """
    if rotation is None:
        rot = np.eye(3)
    else:
        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape == (3,):
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_euler('xyz', rot).as_matrix()

    spacing = np.asarray(spacing, dtype=np.float64)
    semi_axes = np.r_[a, b, c]
    half_extent = np.abs(rot) @ semi_axes
    offset = spacing.copy()
    low = np.ceil(-half_extent - offset)
    high = np.floor(half_extent + offset + 1)
    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    grids = np.mgrid[
        low[0] : high[0] : spacing[0],
        low[1] : high[1] : spacing[1],
        low[2] : high[2] : spacing[2],
    ].astype(np.float64)
    local = np.einsum('ij,j...->i...', rot.T, grids)
    q = (local[0] / a) ** 2 + (local[1] / b) ** 2 + (local[2] / c) ** 2
    return q, low, high, spacing


def test_ellipsoid_rotation_zero_is_identical():
    # Every supported zero-rotation convention must reproduce the original
    # axis-aligned boolean and level-set output exactly.
    for kwargs in (
        {},
        {'rotation': None},
        {'rotation': (0.0, 0.0, 0.0)},
        {'rotation': np.zeros(3)},
        {'rotation': np.eye(3)},
    ):
        assert_array_equal(ellipsoid(3, 5, 7, **kwargs), ellipsoid(3, 5, 7))
        assert_array_equal(
            ellipsoid(3, 5, 7, levelset=True, **kwargs),
            ellipsoid(3, 5, 7, levelset=True),
        )
        # Anisotropic spacing must remain byte-identical as well.
        assert_array_equal(
            ellipsoid(3, 5, 7, spacing=(1.0, 1.5, 2.0), **kwargs),
            ellipsoid(3, 5, 7, spacing=(1.0, 1.5, 2.0)),
        )


def test_ellipsoid_rotation_90deg_x_axis_swap():
    a, b, c = 3, 5, 7
    angle = (np.pi / 2, 0.0, 0.0)

    rotated = ellipsoid(a, b, c, rotation=angle)
    # A 90 degree rotation about the x-axis maps the y semi-axis (length b)
    # onto the z-axis and the z semi-axis (length c) onto the (negative)
    # y-axis, so the rotated solid equals an axis-aligned (a, c, b)
    # ellipsoid.
    expected = ellipsoid(a, c, b)
    assert rotated.shape == expected.shape
    assert_array_equal(rotated, expected)

    # The long-axis tips (semi-axis c along local z) must be present.
    center = np.array(rotated.shape) // 2
    assert rotated[tuple(center)]
    tip_distance = c  # c is now aligned with the world y-axis
    tip_index = center.copy()
    tip_index[1] += int(round(tip_distance))
    assert rotated[tuple(tip_index)]
    tip_index[1] -= 2 * int(round(tip_distance))
    assert rotated[tuple(tip_index)]

    # Explicit rotation matrix and Euler angles must agree.
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert_array_equal(
        ellipsoid(a, b, c, rotation=rot_x),
        ellipsoid(a, b, c, rotation=angle),
    )
    # The level-set form must agree up to floating point precision.
    assert_allclose(
        ellipsoid(a, b, c, rotation=angle, levelset=True),
        ellipsoid(a, c, b, levelset=True),
        atol=1e-12,
    )


def test_ellipsoid_rotation_arbitrary_analytic():
    a, b, c = 3.0, 5.0, 7.0
    angles = np.array([0.37, -1.21, 0.83])
    spacing = (1.0, 1.5, 0.75)

    for levelset in (False, True):
        result = ellipsoid(a, b, c, spacing=spacing, rotation=angles, levelset=levelset)
        q, low, high, sp = _analytic_ellipsoid(a, b, c, angles, spacing)
        assert result.shape == q.shape
        if levelset:
            assert_allclose(result, q - 1.0)
        else:
            assert_array_equal(result, q <= 1.0)

        # The volume must contain every world-axis-aligned half-extent, so
        # no long-axis tip can be clipped.
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler('xyz', angles).as_matrix()
        half_extent = np.abs(rot) @ np.r_[a, b, c]
        coords = [np.arange(low[dim], high[dim], sp[dim]) for dim in range(3)]
        for dim in range(3):
            assert coords[dim].min() <= -half_extent[dim]
            assert coords[dim].max() >= half_extent[dim]
        # One-voxel padding beyond the extents on both sides.
        for dim in range(3):
            assert coords[dim].min() < -half_extent[dim]
            assert coords[dim].max() > half_extent[dim]


def test_ellipsoid_rotation_anisotropic_physical_axes():
    # Rotation acts in physical coordinates: with anisotropic spacing, the
    # rotated principal axis must keep its physical length and the solid
    # stays centered on the middle voxel.
    a, b, c = 2.0, 4.0, 6.0
    spacing = (2.0, 1.0, 0.5)
    angles = (0.0, np.pi / 2, 0.0)  # swap x and z semi-axes

    rotated = ellipsoid(a, b, c, spacing=spacing, rotation=angles)
    expected = ellipsoid(c, b, a, spacing=spacing)
    assert rotated.shape == expected.shape
    assert_array_equal(rotated, expected)

    center = np.array(rotated.shape) // 2
    assert rotated[tuple(center)]

    q, _, _, _ = _analytic_ellipsoid(a, b, c, angles, spacing)
    assert_array_equal(rotated, q <= 1.0)

    # Zero rotation on the anisotropic grid is still the original solid.
    assert_array_equal(
        ellipsoid(a, b, c, spacing=spacing, rotation=(0, 0, 0)),
        ellipsoid(a, b, c, spacing=spacing),
    )


def test_ellipsoid_rotation_invalid_inputs():
    # Wrong number of Euler angles
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=(1.0, 2.0))
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=np.zeros(4))

    # Non-finite angles
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=(np.nan, 0.0, 0.0))
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=(0.0, np.inf, 0.0))

    # Wrong matrix shape
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=np.eye(2))
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=np.zeros((3, 4)))

    # Non-orthogonal matrix
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=np.ones((3, 3)))

    # Orthogonal but improper (reflection, determinant -1)
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=np.diag([1.0, 1.0, -1.0]))

    # Non-finite matrix entries
    bad = np.eye(3)
    bad[0, 1] = np.nan
    with testing.raises(ValueError):
        ellipsoid(2, 2, 2, rotation=bad)


def test_ellipsoid_sign_parameters1():
    with testing.raises(ValueError):
        ellipsoid(-1, 2, 2)


def test_ellipsoid_sign_parameters2():
    with testing.raises(ValueError):
        ellipsoid(0, 2, 2)


def test_ellipsoid_sign_parameters3():
    with testing.raises(ValueError):
        ellipsoid(-3, -2, 2)


def test_ellipsoid_bool():
    test = ellipsoid(2, 2, 2)[1:-1, 1:-1, 1:-1]
    test_anisotropic = ellipsoid(2, 2, 4, spacing=(1.0, 1.0, 2.0))
    test_anisotropic = test_anisotropic[1:-1, 1:-1, 1:-1]

    expected = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    assert_array_equal(test, expected.astype(bool))
    assert_array_equal(test_anisotropic, expected.astype(bool))


def test_ellipsoid_levelset():
    test = ellipsoid(2, 2, 2, levelset=True)[1:-1, 1:-1, 1:-1]
    test_anisotropic = ellipsoid(2, 2, 4, spacing=(1.0, 1.0, 2.0), levelset=True)
    test_anisotropic = test_anisotropic[1:-1, 1:-1, 1:-1]

    expected = np.array(
        [
            [
                [2.0, 1.25, 1.0, 1.25, 2.0],
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [1.0, 0.25, 0.0, 0.25, 1.0],
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [2.0, 1.25, 1.0, 1.25, 2.0],
            ],
            [
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [0.5, -0.25, -0.5, -0.25, 0.5],
                [0.25, -0.5, -0.75, -0.5, 0.25],
                [0.5, -0.25, -0.5, -0.25, 0.5],
                [1.25, 0.5, 0.25, 0.5, 1.25],
            ],
            [
                [1.0, 0.25, 0.0, 0.25, 1.0],
                [0.25, -0.5, -0.75, -0.5, 0.25],
                [0.0, -0.75, -1.0, -0.75, 0.0],
                [0.25, -0.5, -0.75, -0.5, 0.25],
                [1.0, 0.25, 0.0, 0.25, 1.0],
            ],
            [
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [0.5, -0.25, -0.5, -0.25, 0.5],
                [0.25, -0.5, -0.75, -0.5, 0.25],
                [0.5, -0.25, -0.5, -0.25, 0.5],
                [1.25, 0.5, 0.25, 0.5, 1.25],
            ],
            [
                [2.0, 1.25, 1.0, 1.25, 2.0],
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [1.0, 0.25, 0.0, 0.25, 1.0],
                [1.25, 0.5, 0.25, 0.5, 1.25],
                [2.0, 1.25, 1.0, 1.25, 2.0],
            ],
        ]
    )

    assert_allclose(test, expected)
    assert_allclose(test_anisotropic, expected)


def test_ellipsoid_stats():
    # Test comparison values generated by Wolfram Alpha
    vol, surf = ellipsoid_stats(6, 10, 16)
    assert_allclose(1280 * np.pi, vol, atol=1e-6)
    assert_allclose(1383.2828269179892359787, surf, atol=1e-6)

    # Test when a <= b <= c does not hold
    vol, surf = ellipsoid_stats(16, 6, 10)
    assert_allclose(1280 * np.pi, vol, atol=1e-6)
    assert_allclose(1383.2828269179892359787, surf, atol=1e-6)

    # Larger test to ensure reliability over broad range
    vol, surf = ellipsoid_stats(17, 27, 169)
    assert_allclose(103428 * np.pi, vol, atol=1e-4)
    assert_allclose(37426.253635465972897880, surf, atol=1e-6)

    # For equal a, b, c values
    vol, surf = ellipsoid_stats(10, 10, 10)
    assert_allclose(4000 * np.pi / 3, vol, atol=1e-6)
    assert_allclose(400 * np.pi, surf, atol=1e-6)


def test_rect_3d_extent():
    expected = np.array(
        [
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.uint8,
    )
    img = np.zeros((4, 5, 5), dtype=np.uint8)
    start = (0, 0, 2)
    extent = (5, 2, 3)
    pp, rr, cc = rectangle(start, extent=extent, shape=img.shape)
    img[pp, rr, cc] = 1
    assert_array_equal(img, expected)


def test_rect_3d_end():
    expected = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.uint8,
    )
    img = np.zeros((4, 5, 5), dtype=np.uint8)
    start = (1, 0, 2)
    end = (3, 2, 3)
    pp, rr, cc = rectangle(start, end=end, shape=img.shape)
    img[pp, rr, cc] = 1
    assert_array_equal(img, expected)
