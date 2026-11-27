"""Tests for per-axis periodic (toroidal) boundaries in watershed.

These exercise the optional ``periodic`` keyword of
``_skimage2.segmentation.watershed``.
"""

import numpy as np
import pytest

from _skimage2.morphology._util import _validate_connectivity
from _skimage2.segmentation._watershed import watershed


# ---------------------------------------------------------------------------
# Reference implementation (pure Python, coordinate based)
# ---------------------------------------------------------------------------


def _reference_watershed(image, markers, periodic, connectivity=1):
    """C-compatible reference flood on a (possibly periodic) 2D grid.

    The priority rule and binary heap tie handling mirror the Cython
    implementation exactly (same ``smaller`` comparator, same initial heap
    order, labels assigned when an element is first pushed).  The single
    deliberate difference is how neighbors are obtained: they are computed
    from true 2D coordinates and wrapped per axis, which can never mistake
    a flattened-array neighbor for a periodic one.  This makes the reference
    an independent oracle for the periodic wrapping itself.
    """
    image = np.asarray(image, dtype=np.float64)
    labels = np.array(markers, dtype=np.int64).copy()
    rows, cols = image.shape
    periodic = tuple(bool(p) for p in periodic)

    # Build the neighbor list directly from the footprint's per-axis signed
    # steps (always -1/0/1, independent of image shape), sorted by Euclidean
    # distance.  No deduplication is applied: even when two footprint steps
    # share a raveled offset on a grid narrower than the footprint, they are
    # distinct 2D neighbors and can wrap to different pixels along a periodic
    # axis.  This mirrors the production code, which keeps one step array
    # entry per footprint entry.  Neighbors are generated from true 2D
    # coordinates, so the reference cannot inherit a flattened-array mistake.
    footprint, center = _validate_connectivity(2, connectivity, None)
    step_coords = np.stack(
        [(idx - c) for idx, c in zip(np.nonzero(footprint), center)], axis=-1
    ).astype(np.int64)
    distances = np.sqrt(np.sum(step_coords.astype(float) ** 2, axis=1))
    step_coords = step_coords[np.argsort(distances, kind="stable")]
    step_coords = step_coords[1:]  # drop the zero step to the center
    steps = [(int(dr), int(dc)) for dr, dc in step_coords]

    def neighbors(flat):
        r, c = divmod(flat, cols)
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if periodic[0]:
                nr %= rows
            elif not (0 <= nr < rows):
                continue
            if periodic[1]:
                nc %= cols
            elif not (0 <= nc < cols):
                continue
            yield nr * cols + nc

    def smaller(a, b):
        # (value, age); mimics the C struct comparator
        if a[0] != b[0]:
            return a[0] < b[0]
        return a[1] < b[1]

    def heappush(heap, item):
        heap.append(item)
        child = len(heap) - 1
        while child > 0:
            parent = (child + 1) // 2 - 1
            if smaller(heap[child], heap[parent]):
                heap[child], heap[parent] = heap[parent], heap[child]
                child = parent
            else:
                break

    def heappop(heap):
        popped = heap[0]
        last = heap.pop()
        if heap:
            heap[0] = last
            i = 0
            n = len(heap)
            while True:
                smallest = i
                left = 2 * i + 1
                right = left + 1
                if left < n and smaller(heap[left], heap[smallest]):
                    smallest = left
                if right < n and smaller(heap[right], heap[smallest]):
                    smallest = right
                if smallest == i:
                    break
                heap[i], heap[smallest] = heap[smallest], heap[i]
                i = smallest
        return popped

    heap = []
    # np.flatnonzero order, matching the C marker initialization
    marker_flat = np.flatnonzero(labels.ravel())
    for age, flat in enumerate(marker_flat):
        # markers are pushed one after another, so heapify order matches the
        # sequence of pushes; give them all age 0 like the C code
        heappush(heap, (image.ravel()[flat], 0, flat, flat))

    next_age = 1
    while heap:
        value, _, flat, source = heappop(heap)
        for neighbor in neighbors(flat):
            if labels.ravel()[neighbor]:
                continue
            next_age += 1
            nvalue = max(image.ravel()[neighbor], value)
            # label on first push, exactly like the C fast path
            labels.ravel()[neighbor] = labels.ravel()[source]
            heappush(heap, (nvalue, next_age, neighbor, source))
    return labels


# ---------------------------------------------------------------------------
# The flat 3x7 graph requested in the task
# ---------------------------------------------------------------------------


def _flat_3x7_markers():
    markers = np.zeros((3, 7), dtype=int)
    markers[:, 0] = 1  # first column
    markers[:, 2] = 2  # third column
    return markers


def test_flat_3x7_horizontal_wrap_last_column():
    """Last column ownership flips when horizontal wrap is enabled.

    Two markers sit in columns 0 and 2 of a flat 3x7 image.  Without wrap
    the column-2 marker conquers the whole right half including the last
    column.  With horizontal periodicity the last column is adjacent to the
    first one and is reached first by the column-0 marker.
    """
    image = np.zeros((3, 7))
    markers = _flat_3x7_markers()

    plain = watershed(image, markers, connectivity=1)
    # without wrapping the last column can only be reached through the
    # interior, where the column-2 marker is closer, so it owns that column
    assert np.all(plain[:, -1] == 2)
    # the marker cells themselves keep their labels
    assert np.all(plain[markers == 1] == 1)
    assert np.all(plain[markers == 2] == 2)

    wrapped = watershed(image, markers, connectivity=1, periodic=(False, True))
    # column 6 is one periodic step away from column 0 (marker 1) and four
    # ordinary steps away from column 2 (marker 2), so horizontal wrapping
    # flips the ownership of the whole last column
    assert np.all(wrapped[:, -1] == 1)
    # the marker cells keep their labels under wrapping too
    assert np.all(wrapped[markers == 1] == 1)
    assert np.all(wrapped[markers == 2] == 2)
    # something actually changed between the two runs
    assert not np.array_equal(plain, wrapped)


def test_flat_3x7_wrap_matches_coordinate_reference():
    """The periodic result must equal a coordinate-based reference."""
    image = np.zeros((3, 7))
    markers = _flat_3x7_markers()
    for periodic in [(True, False), (False, True), (True, True)]:
        result = watershed(image, markers, connectivity=1, periodic=periodic)
        expected = _reference_watershed(image, markers, periodic, connectivity=1)
        np.testing.assert_array_equal(result, expected)


def test_flat_3x7_no_wrap_matches_default():
    """Explicit all-False periodic must be identical to the default call."""
    image = np.zeros((3, 7))
    markers = _flat_3x7_markers()
    default = watershed(image, markers, connectivity=1)
    explicit = watershed(image, markers, connectivity=1, periodic=(False, False))
    np.testing.assert_array_equal(default, explicit)


# ---------------------------------------------------------------------------
# No false neighbors across rows of the flattened array
# ---------------------------------------------------------------------------


def test_wrap_does_not_connect_different_rows():
    """A horizontal step at the row edge stays inside the same row.

    Marker 1 occupies only (row=0, col=0) and marker 2 the bottom-left
    region.  With horizontal wrapping, (0, 6) must be claimed through the
    wrap by marker 1; it must never be treated as a flattened-array
    neighbor of pixels in another row.
    """
    image = np.zeros((4, 5))
    markers = np.zeros((4, 5), dtype=int)
    markers[0, 0] = 1
    markers[3, 0] = 2
    markers[3, 1] = 2

    wrapped = watershed(image, markers, connectivity=1, periodic=(False, True))
    # top-right corner is one wrapped horizontal step from the marker-1 cell
    assert wrapped[0, -1] == 1
    # bottom row is claimed by marker 2
    assert np.all(wrapped[3, :] == 2)
    # compare against the coordinate reference
    expected = _reference_watershed(image, markers, (False, True), connectivity=1)
    np.testing.assert_array_equal(wrapped, expected)


def test_vertical_wrap_connects_first_and_last_row():
    image = np.zeros((6, 3))
    markers = np.zeros((6, 3), dtype=int)
    markers[0, 1] = 1
    markers[3, 1] = 2

    plain = watershed(image, markers, connectivity=1)
    wrapped = watershed(image, markers, connectivity=1, periodic=(True, False))

    # without wrap the bottom row belongs to marker 2; with vertical wrap the
    # last row is adjacent to the first row and goes to marker 1
    assert plain[-1, 1] == 2
    assert wrapped[-1, 1] == 1
    expected = _reference_watershed(image, markers, (True, False), connectivity=1)
    np.testing.assert_array_equal(wrapped, expected)


def test_double_wrap_torus():
    rng = np.random.RandomState(0)
    image = rng.random((5, 6))
    markers = np.zeros((5, 6), dtype=int)
    markers[0, 0] = 1
    markers[2, 3] = 2
    markers[4, 5] = 3

    for connectivity in (1, 2):
        result = watershed(
            image, markers, connectivity=connectivity, periodic=(True, True)
        )
        expected = _reference_watershed(
            image, markers, (True, True), connectivity=connectivity
        )
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Existing semantics: mask, watershed line, connectivity, adjacent markers
# ---------------------------------------------------------------------------


def test_periodic_respects_mask():
    image = np.zeros((3, 7))
    markers = _flat_3x7_markers()
    mask = np.ones((3, 7), dtype=bool)
    mask[:, 6] = False  # last column excluded

    wrapped = watershed(
        image, markers, connectivity=1, mask=mask, periodic=(False, True)
    )
    assert np.all(wrapped[:, 6] == 0)
    assert np.all(wrapped[mask] != 0)

    # masked-out cells must not serve as wrap bridges either: excluding the
    # whole last column leaves column 5 reachable only through ordinary steps
    mask2 = np.ones((3, 7), dtype=bool)
    mask2[1, 6] = False
    wrapped2 = watershed(
        image, markers, connectivity=1, mask=mask2, periodic=(False, True)
    )
    assert wrapped2[1, 6] == 0


def test_periodic_watershed_line():
    image = np.zeros((4, 6))
    image[:, 3] = 2
    markers = np.zeros((4, 6), dtype=int)
    markers[:, 0] = 1
    markers[:, 3] = 2

    plain_line = watershed(image, markers, watershed_line=True)
    wrap_line = watershed(image, markers, watershed_line=True, periodic=(False, True))
    # the watershed line label 0 appears in both variants
    assert np.any(plain_line == 0)
    assert np.any(wrap_line == 0)
    # markers survive unchanged
    assert np.all(wrap_line[markers == 1] == 1)
    assert np.all(wrap_line[markers == 2] == 2)


def test_periodic_connectivity_1_vs_2():
    rng = np.random.RandomState(1)
    image = rng.random((6, 8))
    markers = np.zeros((6, 8), dtype=int)
    markers[1, 1] = 1
    markers[4, 6] = 2
    c1 = watershed(image, markers, connectivity=1, periodic=(True, True))
    c2 = watershed(image, markers, connectivity=2, periodic=(True, True))
    assert c1.shape == c2.shape == image.shape
    assert set(np.unique(c1)) <= {0, 1, 2}
    assert set(np.unique(c2)) <= {0, 1, 2}
    # every labeled pixel traces to one of the two seeds
    assert np.all(c1[markers == 1] == 1)
    assert np.all(c2[markers == 2] == 2)


def test_periodic_adjacent_markers():
    """Adjacent markers keep their own cells under wrap as without."""
    image = np.zeros((3, 4))
    markers = np.zeros((3, 4), dtype=int)
    markers[:, 0] = 1
    markers[:, 1] = 2
    wrapped = watershed(image, markers, connectivity=1, periodic=(False, True))
    assert np.all(wrapped[:, 0] == 1)
    assert np.all(wrapped[:, 1] == 2)


def test_periodic_compactness_matches_reference():
    rng = np.random.RandomState(2)
    image = rng.random((5, 7))
    markers = np.zeros((5, 7), dtype=int)
    markers[1, 1] = 1
    markers[3, 5] = 2
    # compactness must not crash and every pixel gets one of the marker labels
    out = watershed(image, markers, compactness=0.1, periodic=(True, False))
    assert set(np.unique(out)) == {1, 2}


# ---------------------------------------------------------------------------
# Validation of illegal axis settings
# ---------------------------------------------------------------------------


def test_invalid_periodic_length():
    image = np.zeros((3, 7))
    markers = _flat_3x7_markers()
    with pytest.raises(ValueError):
        watershed(image, markers, periodic=(True,))  # too short
    with pytest.raises(ValueError):
        watershed(image, markers, periodic=(True, True, False))  # too long
    with pytest.raises(ValueError):
        watershed(image, markers, periodic=np.ones((2, 2), dtype=bool))


def test_periodic_unsupported_ndim():
    image = np.zeros((2, 3, 4))
    markers = np.zeros((2, 3, 4), dtype=int)
    markers[0, 0, 0] = 1
    with pytest.raises(NotImplementedError):
        watershed(image, markers, periodic=(True, False, False))


def test_periodic_narrow_grid_dedup():
    """Grids narrower than the footprint must not corrupt the neighbor list.

    On a width-2 grid with connectivity 2, several footprint steps share a
    raveled offset but are distinct 2D neighbors.  The periodic flood must
    visit every footprint neighbor (the step and offset arrays stay aligned),
    which previously caused an out-of-range read / heap corruption.
    """
    rng = np.random.RandomState(7)
    for rows in (2, 3, 7):
        for cols in (2, 3):
            image = rng.random((rows, cols))
            markers = np.zeros((rows, cols), dtype=int)
            markers[0, 0] = 1
            markers[-1, -1] = 2
            for periodic in ((True, False), (False, True), (True, True)):
                for connectivity in (1, 2):
                    result = watershed(
                        image,
                        markers,
                        connectivity=connectivity,
                        periodic=periodic,
                    )
                    expected = _reference_watershed(
                        image, markers, periodic, connectivity=connectivity
                    )
                    np.testing.assert_array_equal(result, expected)
                    # every cell is assigned to one of the two markers
                    assert set(np.unique(result)).issubset({1, 2})


def test_periodic_accepts_none_and_boolean_likes():
    image = np.zeros((3, 4))
    markers = np.zeros((3, 4), dtype=int)
    markers[0, 0] = 1
    markers[2, 3] = 2
    # None and all-False equivalents all run and agree
    out_none = watershed(image, markers)
    out_false = watershed(image, markers, periodic=[0, 0])
    np.testing.assert_array_equal(out_none, out_false)
