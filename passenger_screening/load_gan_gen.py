import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import numpy as np
import time
import glob
import seaborn as sn

output_base = 'gan_output'
output_base = 'gan_tf_output'

row, col = 4, 4
gs = gridspec.GridSpec(4, 4, wspace=1e-2, hspace=1e-30)
sn.plt.ion()
fig = sn.plt.figure(figsize=(8, 8), edgecolor='black', facecolor='black')
axs = []
CMAP = 'hot'

w, h = 512, 660
#w, h = 32, 41

def plot_batch(path):
  data = np.load(path)
  data = np.reshape(data, (w, h, data.shape[0]))
  for i in range(data.shape[2]):
    if len(axs) < data.shape[2]:
      axs.append(fig.add_subplot(gs[i%(row*col)]))
      axs[-1].set_axis_off() 
    #print(data[i])
    axs[i].imshow(data[:,:,i])#, cmap=CMAP)
  #axs[0].set_title(path)
  fig.canvas.draw()

def update_once():
  for i, path in enumerate(glob.iglob('{}/*'.format(output_base))):
    print(path)
    #if not path.endswith('0.npy'): continue
    #plot_batch(path)
    plot_one(path, i)

def plot_one(path, i=0):
    data = np.load(path)
    data = np.reshape(data, (w, h, data.shape[0]))
    total = row*col
    if len(axs) < total:
      axs.append(fig.add_subplot(gs[i%total]))
      axs[-1].set_axis_off() 
    #print(data[i])
    axs[i%total].imshow(np.squeeze(data), cmap=CMAP)
    fig.canvas.draw()

running = True

try:
  print('='*20, output_base, '='*20)
  while running:
    #plot_batch()
    update_once()
except KeyboardInterrupt:
  running = False

