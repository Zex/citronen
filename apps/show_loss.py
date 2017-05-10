import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pylab
import re
import sys

def get_accu(line):
  # score: 0.08709494
  accu = re.search('score:[0-9]*\.[0-9]+', line)
  if accu is None:
      return None
  accu = accu.group().split(':')[1]
  return float(accu)

def get_loss(line):
  #Iteration 83, loss = 0.08709494
  line = line.strip('\n')
  iteration, loss = re.search('\d+', line), re.search('loss = [0-9]*\.[0-9]+', line)
  if iteration is not None and loss is not None:
      iteration, loss = iteration.group(), loss.group()
      loss = loss.split(' ')[2]
  return float(loss)

def pipe_csv(ax, fig):
  losses, accs, cates, sparses = [], [], [], []
  while True:
      line = sys.stdin.readline()
      if line is None or len(line) == 0:
          break
      loss, acc, _, sparse = line.split(',')
      losses.append(float(loss))
      accs.append(float(acc))
      sparses.append(float(sparse))
      pylab.plot(np.arange(len(losses)), losses, '.', color='blue', markerfacecolor='blue')
      pylab.plot(np.arange(len(accs)), accs, '.', color='red', markerfacecolor='red')
      pylab.plot(np.arange(len(sparses)), sparses, '.', color='yellow', markerfacecolor='yellow')
      ax.legend(['loss', 'acc', 'sparse_cate_acc'], loc='upper left')
      fig.canvas.draw()

if __name__ == '__main__':
  plt.ion()
  fig = plt.figure(figsize=(6, 4), facecolor='darkgray', edgecolor='black')
  ax = fig.add_subplot(111, facecolor='black')
  ax.autoscale(True)
  plt.title('profile')
  plt.ylabel('metrics')
  plt.xlabel('steps')
  fig.show()
  #pipe_stdin_loss()
  #pipe_stdin_accu()
  pipe_csv(ax, fig)

