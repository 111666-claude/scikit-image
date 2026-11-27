import numpy as np
from scipy.special import elliprg


def _rotation_matrix(rotation):
    """Validate a rotation input and return a 3x3 rotation matrix.

    See the ``rotation`` parameter description of :func:`ellipsoid` for the
    supported conventions.
    """
    if rotation is None:
        return np.eye(3)

    arr = np.asarray(rotation, dtype=np.float64)

    if arr.ndim == 1:
        # Euler angles: extrinsic rotations about the fixed world x-, y- and
        # z-axis, applied in that order (equivalent to
        # scipy.spatial.transform.Rotation.from_euler('xyz', angles)).
        if arr.shape != (3,):
            raise ValueError(
                "rotation given as Euler angles must contain exactly three "
                f"values (x, y, z), but got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("rotation angles must all be finite.")
        alpha, beta, gamma = arr
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_a, -sin_a],
                [0.0, sin_a, cos_a],
            ]
        )
        rot_y = np.array(
            [
                [cos_b, 0.0, sin_b],
                [0.0, 1.0, 0.0],
                [-sin_b, 0.0, cos_b],
            ]
        )
        rot_z = np.array(
            [
                [cos_g, -sin_g, 0.0],
                [sin_g, cos_g, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return rot_z @ rot_y @ rot_x

    # Explicit 3x3 rotation matrix.
    if arr.shape != (3, 3):
        raise ValueError(
            f"rotation matrix must have shape (3, 3), but got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("rotation matrix must only contain finite values.")
    if not np.allclose(arr.T @ arr, np.eye(3), atol=1e-8):
        raise ValueError("rotation matrix must be orthogonal (R.T @ R == I).")
    if not np.isclose(np.linalg.det(arr), 1.0, atol=1e-8):
        raise ValueError(
            "rotation matrix must have determinant +1 (proper rotation); "
            "a determinant of -1 describes a reflection, which is not "
            "supported."
        )
    return arr


def ellipsoid(
    a,
    b,
    c,
    spacing=(1.0, 1.0, 1.0),
    levelset=False,
    rotation=None,
):
    """Generate ellipsoid for given semi-axis lengths.

    The respective semi-axis lengths are given along three dimensions in
    Cartesian coordinates. Each dimension may use a different grid spacing.

    Parameters
    ----------
    a : float
        Length of semi-axis along x-axis.
    b : float
        Length of semi-axis along y-axis.
    c : float
        Length of semi-axis along z-axis.
    spacing : 3-tuple of floats
        Grid spacing in three spatial dimensions.
    levelset : bool
        If True, returns the level set for this ellipsoid (signed level
        set about zero, with positive denoting interior) as np.float64.
        False returns a binarized version of said level set.
    rotation : None, 3-tuple of float, or 3x3 array_like, optional
        Rotation applied to the ellipsoid before voxelization. ``None`` (the
        default) yields an axis-aligned ellipsoid and is fully equivalent to
        not passing a rotation. Two conventions are supported:

        - **Euler angles** given as a 3-tuple ``(x, y, z)`` of angles in
          radians. These are extrinsic rotations about the fixed world x-,
          y- and z-axis, applied in that order (x first, z last). This is
          equivalent to
          ``scipy.spatial.transform.Rotation.from_euler('xyz', angles)``.
          For example, ``rotation=(np.pi / 2, 0, 0)`` rotates by 90 degrees
          about the world x-axis.
        - **Rotation matrix** given as a 3x3 array. The matrix ``R`` maps
          ellipsoid-local (principal-axis) coordinates ``p`` to world
          coordinates as ``p_world = R @ p_local``; equivalently, the
          rotated ellipsoid is the set of points ``p`` for which
          ``(R.T @ p)`` lies inside the axis-aligned ellipsoid. The matrix
          must be orthogonal with determinant +1.

        Rotation acts in physical coordinates, so anisotropic ``spacing``
        does not distort the physical semi-axes or move the center. The
        output volume is enlarged as needed to contain the rotated solid.

    Returns
    -------
    ellipsoid : (M, N, P) array
        Ellipsoid centered in a correctly sized array for given `spacing`
        and `rotation`. Boolean dtype unless `levelset=True`, in which case a
        float array is returned with the level set above 0.0 representing
        the ellipsoid.

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    rot = _rotation_matrix(rotation)
    semi_axes = np.r_[a, b, c]

    offset = np.r_[1, 1, 1] * np.r_[spacing]

    # Physical half-extent of the rotated ellipsoid along each world axis.
    # The columns of `rot` are the world-space directions of the ellipsoid's
    # principal axes; each axis projects its full semi-axis length onto the
    # world axes. Using these extents guarantees that no long-axis tip is
    # clipped by the output volume.
    half_extent = np.abs(rot) @ semi_axes

    # Calculate limits, and ensure output volume is odd & symmetric
    low = np.ceil(-half_extent - offset)
    high = np.floor(half_extent + offset + 1)

    for dim in range(3):
        if (high[dim] - low[dim]) % 2 == 0:
            low[dim] -= 1
        num = np.arange(low[dim], high[dim], spacing[dim])
        if 0 not in num:
            low[dim] -= np.max(num[num < 0])

    # Generate (anisotropic) spatial grid
    x, y, z = np.mgrid[
        low[0] : high[0] : spacing[0],
        low[1] : high[1] : spacing[1],
        low[2] : high[2] : spacing[2],
    ]

    # Transform world coordinates into the ellipsoid's principal-axis frame.
    # For the identity rotation this reduces to the original computation
    # bit-for-bit, keeping zero-rotation output unchanged.
    coords = np.stack([x, y, z], axis=0).astype(np.float64, copy=False)
    local = np.einsum('ij,j...->i...', rot.T, coords)
    xl, yl, zl = local[0], local[1], local[2]

    if not levelset:
        arr = ((xl / float(a)) ** 2 + (yl / float(b)) ** 2 + (zl / float(c)) ** 2) <= 1
    else:
        arr = ((xl / float(a)) ** 2 + (yl / float(b)) ** 2 + (zl / float(c)) ** 2) - 1

    return arr


def ellipsoid_stats(a, b, c):
    """Calculate analytical volume and surface area of an ellipsoid.

    The surface area of an ellipsoid is given by

    .. math:: S=4\\pi b c R_G\\!\\left(1, \\frac{a^2}{b^2}, \\frac{a^2}{c^2}\\right)

    where :math:`R_G` is Carlson's completely symmetric elliptic integral of
    the second kind [1]_. The latter is implemented as
    :py:func:`scipy.special.elliprg`.

    Parameters
    ----------
    a : float
        Length of semi-axis along x-axis.
    b : float
        Length of semi-axis along y-axis.
    c : float
        Length of semi-axis along z-axis.

    Returns
    -------
    vol : float
        Calculated volume of ellipsoid.
    surf : float
        Calculated surface area.

    References
    ----------
    .. [1] Paul Masson (2020). Surface Area of an Ellipsoid.
           https://analyticphysics.com/Mathematical%20Methods/Surface%20Area%20of%20an%20Ellipsoid.htm

    """
    if (a <= 0) or (b <= 0) or (c <= 0):
        raise ValueError('Parameters a, b, and c must all be > 0')

    # Volume
    vol = 4 / 3.0 * np.pi * a * b * c

    # Surface area
    surf = 3 * vol * elliprg(1 / a**2, 1 / b**2, 1 / c**2)

    return vol, surf
