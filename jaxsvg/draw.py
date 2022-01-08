import functools

import jax
import jax.numpy as np

from jaxsvg import util


def draw_path(
    bezier_parameters,
    dim=256,
    sampling_res=256,
    just_outline=False,
):
  """Draw a filled path mask with anti-aliasing according to Bezier curves.

  Arguments:
    bezier_parameters: Bezier curve anchor points, shape [num_curves, 3, 2]
    dim: the x and y dimensions of the canvas
    sampling_res: the sampling resolution for polyline fill and point stroke
  
  Returns:
    A float32 alpha mask of the parameterized filled path, shape [dim, dim]
  """
  start = bezier_parameters[:, 0]
  c1 = bezier_parameters[:, 1]
  c2 = bezier_parameters[:, 2]
  end = np.roll(start, -1, axis=-2)
  points = util.cubic(start, end, c1, c2, res=sampling_res)
  points = np.clip(points, 0, 1)
  points = np.reshape(points, [-1, 2])
  x_loc = util.get_raytrace_points((points + (1 / dim / 2)), dim=dim)
  y_loc = util.get_raytrace_points(
      (np.roll(points, 1, axis=-1) + (1 / dim / 2)), dim=dim)
  idxs_x = np.sort(np.where(
      x_loc,
      np.tile(np.arange(x_loc.shape[0])[:, np.newaxis], [1, x_loc.shape[1]]),
      np.ones(x_loc.shape, dtype=np.int32) * -1),
                   axis=0)[::-1][:4]
  idxs_y = np.sort(np.where(
      y_loc,
      np.tile(np.arange(y_loc.shape[0])[:, np.newaxis], [1, y_loc.shape[1]]),
      np.ones(y_loc.shape, dtype=np.int32) * -1),
                   axis=0)[::-1][:4]

  sorted_x = np.sort(x_loc[idxs_x,
                           np.tile(np.arange(dim)[np.newaxis], [4, 1])],
                     axis=0)[::-1]
  sorted_y = np.sort(y_loc[idxs_y,
                           np.tile(np.arange(dim)[np.newaxis], [4, 1])],
                     axis=0)[::-1]

  def draw_outline_from_raytrace(sorted_loc):
    sorted_loc = sorted_loc - (1 / dim / 2)
    intersects = np.where(sorted_loc > 0, sorted_loc,
                          np.ones_like(sorted_loc) * -1)
    drawn_loc = np.abs((np.arange(dim) / dim)[:, np.newaxis, np.newaxis] -
                       intersects).min(axis=1)
    return drawn_loc

  stroke_width = 2
  threshold = stroke_width / dim
  x_thresh = draw_outline_from_raytrace(sorted_x)
  y_thresh = draw_outline_from_raytrace(sorted_y).T
  distance_pixels = np.min(np.stack([x_thresh, y_thresh]), axis=0)
  distance_pixels = np.where(distance_pixels > threshold,
                             np.ones_like(distance_pixels) * threshold,
                             distance_pixels)
  mask = util.fill_shape(sorted_x)
  outline = ((threshold) - distance_pixels) / threshold
  if just_outline:
    return outline
  mask = mask + outline
  mask = np.where(mask > 1, np.ones_like(mask), mask)
  return mask


def compose_paths(bezier_parameters,
                  color_parameters,
                  background_color=np.array([1, 1, 1]),
                  dim=256,
                  sampling_res=256):
  """Draw many filled paths and compose them together with color and implicit
  z-ordering.

  Arguments:
    bezier_parameters: Bezier curve anchor points per path,
      shape [num_paths, num_curves, 3, 2]
    color_parameters: RGBA value per path, shape [num_paths, 4]
    dim: the x and y dimensions of the canvas
    sampling_res: the sampling resolution for polyline fill and point stroke
  """
  draw_path(bezier_parameters[0], dim=dim)
  path_masks = jax.vmap(
      functools.partial(draw_path, dim=dim,
                        sampling_res=sampling_res))(bezier_parameters)
  colored_masks = path_masks[...,
                             np.newaxis] * color_parameters[:, np.newaxis,
                                                            np.newaxis, :3]
  canvas = np.zeros([dim, dim, 3], dtype=np.float32)
  remaining_alpha = np.ones([dim, dim], dtype=np.float32)
  num_paths = colored_masks.shape[0]
  for i in range(num_paths):
    alpha = color_parameters[i, 3]
    effective_alpha = remaining_alpha * path_masks[i] * alpha
    remaining_alpha -= effective_alpha
    canvas += effective_alpha[...,
                              np.newaxis] * color_parameters[i, :3][np.newaxis,
                                                                    np.newaxis]
  canvas += np.ones([dim, dim, 1]) * background_color[
      np.newaxis, np.newaxis] * remaining_alpha[..., np.newaxis]
  return canvas
