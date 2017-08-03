import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import numpy as np
import time
import glob
import seaborn as sn

output_base = 'gan_output'
gs = gridspec.GridSpec(4, 4, wspace=1e-2, hspace=1e-20)
sn.plt.ion()
fig = sn.plt.figure(figsize=(8, 8), edgecolor='black', facecolor='black')
axs = []
CMAP = 'hot'

def plot_batch(data):
  data = np.reshape(data, (512, 660, 16))
  for i in range(16):
    if len(axs) < 16:
      axs.append(fig.add_subplot(gs[i%16]))
      axs[-1].set_axis_off() 
    print(data[:,:,i])
    #data[:,:,i][np.where(data[:,:,i] < 10000)] = 0
    axs[i].imshow(data[:,:,i], cmap=CMAP)
    fig.canvas.draw()

def update_once():
  for i, path in enumerate(glob.iglob('{}/*'.format(output_base))):
    print(path)
    if '/0.npy' in path: continue
    data = np.load(path)
    plot_batch(data)

running = True

while running:
  try:
    #plot_batch()
    update_once()
  except KeyboardInterrupt:
    running = False

