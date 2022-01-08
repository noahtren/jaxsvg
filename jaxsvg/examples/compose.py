import jax.numpy as np
import matplotlib.pyplot as plt

from jaxsvg import draw

if __name__ == "__main__":
  bezier_params = [
      np.array([[[0.25, 0.25], [1.0, 1.0], [0.0, 0.7]]]),
      np.array([[[0.8, 0.5], [0.2, 0.8], [0.6, 0.2]]])
  ]
  movie = []
  movie.append(draw.draw_path(bezier_params[0], just_outline=True))
  movie.append(draw.draw_path(bezier_params[1], just_outline=True))
  imgs = [draw.draw_path(b, just_outline=True) for b in bezier_params]
  movie.append(np.stack(imgs).max(axis=0))
  imgs = [
      draw.draw_path(bezier_params[0], just_outline=False),
      draw.draw_path(bezier_params[1], just_outline=True)
  ]
  movie.append(np.stack(imgs).max(axis=0))
  imgs = [
      draw.draw_path(bezier_params[0], just_outline=False),
      draw.draw_path(bezier_params[1], just_outline=False)
  ]
  movie.append(np.stack(imgs).max(axis=0))
  fig, axes = plt.subplots(1, 5)
  for i in range(5):
    axes[i].imshow(movie[i])
  plt.show()