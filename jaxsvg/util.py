import jax
import jax.numpy as np


def line(a, b, res):
  @jax.vmap
  def vline(a, b):
    return np.linspace(a, b, num=res)

  return vline(a, b)


@jax.vmap
def lerp_line_pair(a, b):
  coeff = np.linspace(0, 1, num=a.shape[0])[..., np.newaxis]
  return a * (1 - coeff) + b * coeff


def cubic(start, end, c1, c2, res):
  cl1 = lerp_line_pair(line(start, c1, res), line(c1, c2, res))
  cl2 = lerp_line_pair(line(c1, c2, res), line(c2, end, res))
  return lerp_line_pair(cl1, cl2)


def draw_outline(points, stroke_width=1, dim=128):
  """Returns:
    threshold_dists: [num_lines, t, 2]
  """
  pointgrid = np.stack(np.meshgrid(np.arange(dim), np.arange(dim)),
                       axis=-1) / dim
  point_dists = ((points[..., np.newaxis, np.newaxis, :] -
                  pointgrid[np.newaxis, np.newaxis])**2).sum(axis=-1)**(1 / 2)
  point_dists = point_dists.min(axis=[0, 1])
  threshold = (stroke_width * 2 / dim)
  threshold_dists = np.where(point_dists < threshold,
                             (threshold - point_dists) / threshold,
                             np.zeros_like(point_dists))
  return threshold_dists


def get_raytrace_points(points, dim=128):
  p1 = points
  p2 = np.roll(p1, -1, axis=0)
  delta = p2 - p1
  rays = np.linspace(0, 1, num=dim)
  p_y = (rays[np.newaxis] -
         p1[:, 0][:, np.newaxis]) / (delta[:, 0][:, np.newaxis] + 1e-12)
  p_y = np.where(np.logical_and(p_y < 1, p_y > 0), p_y, np.zeros_like(p_y))
  loc = p_y * delta[:, 1][:, np.newaxis] + p1[:, 1][:, np.newaxis]
  loc = np.where(np.logical_and(p_y < 1, p_y > 0), loc, np.zeros_like(loc))
  return loc


def fill_shape(sorted_xloc):
  img_res = sorted_xloc.shape[1]
  sorted_xloc_end, sorted_xloc_start = sorted_xloc[:-1], sorted_xloc[1:]
  points = np.linspace(0, 1, img_res)
  segments = np.logical_and(
      points[np.newaxis, np.newaxis] > sorted_xloc_start[..., np.newaxis],
      points[np.newaxis, np.newaxis] < sorted_xloc_end[..., np.newaxis])
  shape = segments[::2].sum(axis=0).astype(np.float32)
  return shape.T
