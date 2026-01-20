import jax
import jax.numpy as np
from jax import Array


def line(a: Array, b: Array, res: int) -> Array:
    """Linearly interpolate between points a and b."""

    @jax.vmap
    def vline(a: Array, b: Array) -> Array:
        return np.linspace(a, b, num=res)

    return vline(a, b)


@jax.vmap
def lerp_line_pair(a: Array, b: Array) -> Array:
    """Linearly interpolate between two lines."""
    coeff = np.linspace(0, 1, num=a.shape[0])[..., np.newaxis]
    return a * (1 - coeff) + b * coeff


def cubic(
    start: Array,
    end: Array,
    c1: Array,
    c2: Array,
    res: int,
) -> Array:
    """Evaluate cubic Bezier curve using de Casteljau's algorithm.

    Arguments:
      start: start points, shape [num_curves, 2]
      end: end points, shape [num_curves, 2]
      c1: first control points, shape [num_curves, 2]
      c2: second control points, shape [num_curves, 2]
      res: number of points to sample along each curve

    Returns:
      Sampled points along curves, shape [num_curves, res, 2]
    """
    cl1 = lerp_line_pair(line(start, c1, res), line(c1, c2, res))
    cl2 = lerp_line_pair(line(c1, c2, res), line(c2, end, res))
    return lerp_line_pair(cl1, cl2)


def draw_outline(
    points: Array,
    stroke_width: int = 1,
    dim: int = 128,
) -> Array:
    """Compute distance-based outline from sampled points.

    Arguments:
      points: sampled curve points, shape [num_curves, num_points, 2]
      stroke_width: width of the stroke in pixels
      dim: canvas dimension

    Returns:
      Thresholded distance field, shape [dim, dim]
    """
    pointgrid = np.stack(np.meshgrid(np.arange(dim), np.arange(dim)), axis=-1) / dim
    point_dists = (
        (points[..., np.newaxis, np.newaxis, :] - pointgrid[np.newaxis, np.newaxis])
        ** 2
    ).sum(axis=-1) ** (1 / 2)
    point_dists = point_dists.min(axis=[0, 1])
    threshold = stroke_width * 2 / dim
    threshold_dists = np.where(
        point_dists < threshold,
        (threshold - point_dists) / threshold,
        np.zeros_like(point_dists),
    )
    return threshold_dists


def get_raytrace_points(points: Array, dim: int = 128) -> Array:
    """Compute ray-line intersection points for scanline rasterization.

    Arguments:
      points: polyline vertices, shape [num_points, 2]
      dim: number of scanlines (rays)

    Returns:
      Intersection y-coordinates per ray, shape [num_segments, dim]
    """
    p1 = points
    p2 = np.roll(p1, -1, axis=0)
    delta = p2 - p1
    rays = (np.arange(dim) + 0.5) / dim
    denom = delta[:, 0][:, np.newaxis]
    safe_denom = np.where(np.abs(denom) < 1e-8, 1e-8 * np.sign(denom + 1e-12), denom)
    p_y = (rays[np.newaxis] - p1[:, 0][:, np.newaxis]) / safe_denom
    
    valid_intersection = np.logical_and(p_y < 1, p_y >= 0)
    
    p_y = np.where(valid_intersection, p_y, np.zeros_like(p_y))
    loc = p_y * delta[:, 1][:, np.newaxis] + p1[:, 1][:, np.newaxis]
    loc = np.where(valid_intersection, loc, np.zeros_like(loc))
    return loc


def fill_shape(sorted_xloc: Array, height: int) -> Array:
    """Fill shape interior using even-odd rule from sorted x intersections.

    Arguments:
      sorted_xloc: sorted x intersection locations, shape [max_intersections, width]
      height: height of output image

    Returns:
      Filled mask, shape [height, width]
    """
    sorted_xloc_end, sorted_xloc_start = sorted_xloc[:-1], sorted_xloc[1:]
    points = (np.arange(height) + 0.5) / height
    segments = np.logical_and(
        points[np.newaxis, np.newaxis] > sorted_xloc_start[..., np.newaxis],
        points[np.newaxis, np.newaxis] < sorted_xloc_end[..., np.newaxis],
    )
    shape = segments[::2].sum(axis=0).astype(np.float32)
    return shape.T
