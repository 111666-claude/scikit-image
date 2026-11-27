import numpy as np
import scipy.ndimage as ndi

from skimage._shared._warnings import warn_external
from skimage._shared.coord import ensure_spacing


def _default_footprint(min_distance, ndim, p_norm):
    """Create the default neighborhood for `peak_local_max`.

    The footprint only connects coordinates whose Minkowski distance is
    strictly smaller than `min_distance`. Pairs at exactly `min_distance`
    are left for `ensure_spacing`, which is responsible for enforcing the
    inclusive distance bound and its tie-breaking rules.
    """
    # Keep the large pre-filtering neighborhood close to the historical
    # hypercube, but end it immediately before the inclusive distance bound.
    # Tightening the bounding box also makes the common Chebyshev footprint
    # dense, allowing scipy to use its separable maximum-filter path.
    radius = int(np.ceil(min_distance)) - 1
    size = 2 * radius + 1

    # A large non-rectangular footprint makes scipy's generic ND filter path
    # prohibitively expensive. In that case use only the immediate local
    # neighborhood; ``ensure_spacing`` still applies the full distance rule.
    if not np.isinf(p_norm) and size**ndim > 100_000:
        radius = 1
        size = 3

    footprint = np.zeros((size,) * ndim, dtype=bool)

    indices = np.indices(footprint.shape) - radius
    abs_indices = np.abs(indices)

    if np.isinf(p_norm):
        distance = np.max(abs_indices, axis=0)
    elif p_norm == 1:
        distance = np.sum(abs_indices, axis=0)
    elif p_norm == 2:
        distance = np.sqrt(np.sum(indices**2, axis=0))
    else:
        # Scale before exponentiating so that large finite p norms do not
        # overflow while constructing the default footprint.
        max_abs = np.max(abs_indices, axis=0)
        scale = np.where(max_abs == 0, 1, max_abs)
        scaled = abs_indices / scale
        distance = max_abs * np.sum(scaled**p_norm, axis=0) ** (1 / p_norm)

    footprint[distance < min_distance] = True
    return footprint


def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities, kind="stable")
    coord = np.transpose(coord)[idx_maxsort]

    if min_distance > 1:
        coord = ensure_spacing(
            coord, spacing=min_distance, p_norm=p_norm, max_out=num_peaks
        )

    if num_peaks is not None and len(coord) > num_peaks:
        coord = coord[:num_peaks]

    return coord


def _get_peak_mask(image, footprint, threshold, mask=None):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint.size == 1 or image.size == 1:
        return image > threshold

    image_max = ndi.maximum_filter(image, footprint=footprint, mode='nearest')

    out = image == image_max

    # no peak for a trivial image
    image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
    if image_is_trivial:
        out[:] = False
        if mask is not None:
            # isolated pixels in masked area are returned as peaks
            isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
            out[isolated_px] = True

    out &= image > threshold
    return out


def _exclude_border(label, border_width):
    """Set label border values to 0."""
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label


def _get_threshold(image, threshold_abs, threshold_rel):
    """Return the threshold value according to an absolute and a relative
    value.

    """
    threshold = threshold_abs if threshold_abs is not None else image.min()

    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())

    return threshold


def _validate_exclude_border(exclude_border, *, ndim):
    """Return border_width values relative to a min_distance if requested."""

    if isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        border_width = (exclude_border,) * ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image."
            )
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError("`exclude_border` can not be a negative value")
        border_width = exclude_border
    else:
        raise TypeError(
            "`exclude_border` must be int or tuple with the same "
            "length as the dimensionality of the image."
        )

    return border_width


def peak_local_max(
    image,
    *,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=1,
    num_peaks=None,
    footprint=None,
    labels=None,
    num_peaks_per_label=None,
    p_norm=2.0,
):
    """Find peaks in an image as coordinate list.

    Peaks are selected so that their Minkowski distance (see `p_norm`) is at
    least `min_distance`. The default maximum-filter footprint only compares
    positions at a distance strictly smaller than `min_distance`, so peaks
    separated by exactly `min_distance` remain eligible for final selection.

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : float, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`. See also `p_norm`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int or tuple of (int, ...), optional
        Control peak detection close to the border of `image`. By default,
        only peaks exactly on the border are excluded.

        ``0``
            Distance to border has no effect, all peaks are identified.
        positive integer
            Exclude peaks that are within this distance of the border.
        tuple of positive integers
            Same as for a single integer but with different distances for each
            respective dimension.

        The value of `p_norm` has no impact on this border distance.
    num_peaks : int, optional
        If given, maximum number of allowed peaks. When the number of peaks
        exceeds `num_peaks`, return `num_peaks` peaks based on highest peak
        intensity.
    footprint : ndarray of dtype bool, optional
        Binary mask that determines the neighborhood (where ``True``) in which
        a peak must be a local maximum (see *Notes*). If not given, the
        neighborhood is derived from `min_distance` and `p_norm`: it contains
        offsets whose Minkowski distance is strictly smaller than
        `min_distance`. For ``min_distance=1``, the historical full
        three-pixel-wide hypercube is used.
    labels : ndarray of dtype int, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero labels are reserved for background.
    num_peaks_per_label : int, optional
        If given, maximum number of peaks for each label.
    p_norm : float, optional
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance. See also :func:`numpy.linalg.norm`.

    Returns
    -------
    output : ndarray of shape (N, D)
        The coordinates of the peaks. ``N`` denotes the number of peaks and
        ``D`` corresponds to the number of dimensions in `image`.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. Internally, a maximum filter is used for finding
    local maxima. This operation dilates the original image. After comparison
    of the dilated and original images, this function returns the coordinates
    of the peaks where the dilated image equals the original image.

    See also
    --------
    skimage2.feature.corner_peaks

    Examples
    --------
    >>> import skimage2 as ski2
    >>> image = np.array(
    ...     [[1, 0, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 1, 0, 0],
    ...      [0, 0, 3, 0, 2, 0, 0],
    ...      [0, 0, 2, 0, 0, 0, 0],
    ...      [0, 0, 0, 0, 0, 0, 0]]
    ... )

    Find all peaks
    >>> ski2.feature.peak_local_max(image)
    array([[3, 2],
           [3, 4]])

    Peaks exactly 2 pixels apart are both allowed
    >>> ski2.feature.peak_local_max(image, min_distance=2)
    array([[3, 2],
           [3, 4]])

    Allow peaks on the image border
    >>> ski2.feature.peak_local_max(image, exclude_border=0)
    array([[3, 2],
           [3, 4],
           [0, 0]])
    """
    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn_external(
            "When `min_distance < 1`, `peak_local_max` acts as finding "
            "`image > max(threshold_abs, threshold_rel * max(image))`.",
            category=RuntimeWarning,
        )

    border_width = _validate_exclude_border(exclude_border, ndim=image.ndim)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        if min_distance <= 1:
            # Keep the established unit neighborhood (including its
            # plateau handling) for the historical default distance.
            size = int(np.floor(2 * min_distance + 1))
            footprint = np.ones((size,) * image.ndim, dtype=bool)
        else:
            footprint = _default_footprint(
                min_distance=min_distance, ndim=image.ndim, p_norm=p_norm
            )
    else:
        footprint = np.asarray(footprint)

    if labels is None:
        # Non maximum filter
        mask = _get_peak_mask(image, footprint, threshold)

        mask = _exclude_border(mask, border_width)

        # Select highest intensities (num_peaks)
        coordinates = _get_high_intensity_peaks(
            image, mask, num_peaks, min_distance, p_norm
        )

    else:
        _labels = _exclude_border(labels.astype(int, casting="safe"), border_width)

        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks
        labels_peak_coord = []

        for label_idx, roi in enumerate(ndi.find_objects(_labels)):
            if roi is None:
                continue

            # Get roi mask
            label_mask = labels[roi] == label_idx + 1
            # Extract image roi
            img_object = image[roi].copy()
            # Ensure masked values don't affect roi's local peaks
            img_object[np.logical_not(label_mask)] = bg_val

            mask = _get_peak_mask(img_object, footprint, threshold, label_mask)

            coordinates = _get_high_intensity_peaks(
                img_object, mask, num_peaks_per_label, min_distance, p_norm
            )

            # transform coordinates in global image indices space
            for idx, s in enumerate(roi):
                coordinates[:, idx] += s.start

            labels_peak_coord.append(coordinates)

        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, image.ndim), dtype=int)

        if num_peaks is not None and len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(
                image, out, num_peaks, min_distance, p_norm
            )

    return coordinates
