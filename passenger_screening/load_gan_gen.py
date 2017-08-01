import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import numpy as np
import time
import glob
import seaborn as sn

#output_base = 'gan_output'
output_base = 'gan_output_linear'
gs = gridspec.GridSpec(4, 4, wspace=0.05, hspace=0.05)
sn.plt.ion()
fig = sn.plt.figure(figsize=(8, 8), edgecolor='black', facecolor='black')
axs = []
#sn.plt.show()

def plot(x):
  data = np.load(output_base+'/x.npy')
  data = np.reshape(data, (512, 660, 16)).astype(np.int32)
  print(data[:,:,i])
  for l in data[:,:,i]:
    print(l)
  for i in range(16):
    if len(axs) < 16:
      axs.append(fig.add_subplot(gs[i%16]))
      axs[-1].set_axis_off() 
    axs[i].imshow(data[:,:,i])#, cmap='viridis')
    fig.canvas.draw()

def update_once():
  for i, path in enumerate(glob.iglob('{}/*'.format(output_base))):
    print(path)
    data = np.load(path)
    try:
      data = np.reshape(data, (512, 660))#.astype(np.int32)
    except ValueError:
      continue
    print(data)
    if len(axs) < 16:
      axs.append(fig.add_subplot(gs[i%16]))
      axs[-1].set_axis_off() 
    axs[i].imshow(data, cmap='viridis')
    fig.canvas.draw()

while True:
  update_once()
  time.sleep(1)

