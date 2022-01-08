import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import imageio
import jax.numpy as np
import numpy as onp
from jax import random
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

from jaxsvg import draw


def train_step(optimizer, opt_state, p, z):

  def loss_fn(params):
    bezier_parameters = model.apply({'params': params}, z)
    raster = draw.draw_path(bezier_parameters[0][0], sampling_res=32)
    target = np.ones_like(raster)
    loss = ((raster - target)**2).mean()
    metrics = {'raster': raster, 'loss': loss}
    return loss, metrics

  grad, metrics = jax.grad(loss_fn, has_aux=True)(p)
  updates, opt_state = optimizer.update(grad, opt_state)
  p = optax.apply_updates(p, updates)
  return p, opt_state, metrics


class CyclicConv1D(nn.Module):
  wsize: int
  features: int

  @nn.compact
  def __call__(self, x):
    x = np.pad(x, ((0, 0), (0, 0), (self.wsize, self.wsize), (0, 0)),
               mode='wrap')
    x = jax.vmap(
        nn.Conv(features=self.features,
                kernel_size=1 + self.wsize * 2,
                strides=[1],
                padding='valid',
                kernel_init=nn.initializers.normal(1e-4)))(x)
    x = nn.gelu(x)
    x = nn.LayerNorm()(x)
    return x


class ZToBezierParam(nn.Module):
  curves_per_shape: int
  num_shapes: int
  layer_size: int
  num_conv_layers: int

  @nn.compact
  def __call__(self, z):
    batch_size = z.shape[0]
    per_shape_z = np.reshape(
        nn.Dense(features=self.num_shapes * self.layer_size)(z),
        [batch_size, self.num_shapes, self.layer_size])
    curve_starts = np.arange(
        self.curves_per_shape) / (self.curves_per_shape) * 2 * np.pi
    start_coords = np.stack(
        [np.sin(curve_starts), np.cos(curve_starts)], axis=-1)
    top_anchor_points = start_coords + np.stack(
        [np.sin(curve_starts + np.pi / 2),
         np.cos(curve_starts + np.pi / 2)],
        axis=-1)
    bot_anchor_points = start_coords - np.stack(
        [np.sin(curve_starts + np.pi / 2),
         np.cos(curve_starts + np.pi / 2)],
        axis=-1)
    # the following 2-liner is similar to np.dstack. We interleave start, top, and bot
    loc_params = np.stack([
        start_coords, top_anchor_points,
        np.roll(bot_anchor_points, -1, axis=0)
    ],
                          axis=1)
    loc_params = np.reshape(loc_params, [-1, 2])

    x = np.concatenate([
        np.tile(loc_params[np.newaxis, np.newaxis],
                [batch_size, self.num_shapes, 1, 1]),
        np.tile(per_shape_z[:, :, np.newaxis], [1, loc_params.shape[0], 1])
    ],
                       axis=-1)
    # cyclic convolutions along shape parameter coordinate (c) axis: [b, s, c, 2]
    for _ in range(self.num_conv_layers):
      x = CyclicConv1D(features=self.layer_size, wsize=1)(x)

    bezier_parameters = nn.DenseGeneral(features=2, name='shape_bezier')(x)
    bezier_parameters = np.reshape(
        bezier_parameters,
        [batch_size, self.num_shapes, self.curves_per_shape, 3, 2])
    bezier_parameters = nn.sigmoid(
        bezier_parameters / np.sqrt(self.layer_size) +
        np.reshape(loc_params, [batch_size, 1, self.curves_per_shape, 3, 2]))

    return bezier_parameters


if __name__ == "__main__":
  optimizer = optax.adam(1e-3)
  z = np.zeros([1, 512])
  model = ZToBezierParam(curves_per_shape=3,
                         num_shapes=1,
                         layer_size=512,
                         num_conv_layers=5)
  model_state = model.init(random.PRNGKey(0), z)
  params = model_state['params']

  opt_state = optimizer.init(params)
  init_shape, final_shape = None, None

  writer = imageio.get_writer("video.mp4")
  for i in range(32):
    params, opt_state, metrics = train_step(optimizer, opt_state, params, z)
    if i == 0:
      init_shape = metrics['raster']
    elif i == 31:
      final_shape = metrics['raster']
    if i > 1:
      writer.append_data((onp.array(
          np.tile(metrics['raster'][..., np.newaxis], [1, 1, 3]) * 255).astype(
              onp.uint8)))
    print(metrics['loss'])

  writer.close()

  fig, axes = plt.subplots(2)
  axes[0].imshow(init_shape)
  axes[1].imshow(final_shape)
  plt.show()
