import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import pylab
import re
import sys
import time
import argparse

def get_accu(line):
  # score: 0.08709494
  accu = re.search('score:[0-9]*\.[0-9]+', line)
  if accu is None:
      return None
  accu = accu.group().trim('[]')
  return float(accu)

def get_loss(line, emp=10):
  #Iteration 83, loss = 0.08709494
  line = line.strip('\n')
  epoch, loss = re.search('\[[\d]+\]', line), re.search('loss: (-){0,1}[0-9]*\.[0-9]+', line)
  if epoch:
    epoch = epoch.group().strip('[]')
  if loss:
    loss = loss.group().split(' ')[1]
  return epoch, np.round(float(loss)*emp, 4)

def pipe_loss(flow=False, trim_level=None, stage_size=1000, emp=10, window_size=100):
  global fig, ax1, ax2
  trim_nr, losses, accs = 0, [], []
  while True:
    line = sys.stdin.readline()
    if line is None or len(line) == 0:
        break
    epoch, loss = get_loss(line, emp)
    print('epoch:{}, loss:{}'.format(epoch, loss))
    if trim_level and loss > trim_level:
      trim_nr += 1
      continue
    losses.append(loss)

    if flow:
      if len(losses) > window_size:
        sub_losses = losses[-window_size:]
      else:
        sub_losses = losses[:]
      marker = '-'
      ax1.cla()
      ax1.plot(np.arange(len(sub_losses)), sub_losses, marker, color='yellow', markerfacecolor='yellow')
      plt.title('[{}] {}/{} -{}'.format(epoch, len(losses), loss, trim_nr))
      fig.canvas.draw()

    if len(losses) % stage_size != 0:
      continue
    else:
      marker = '-'
      ax2.cla()
      ax2.plot(np.arange(len(losses)), losses, marker, color='blue', markerfacecolor='blue')
      plt.title('[{}] {}/{} -{}'.format(epoch, len(losses), loss, trim_nr))
      fig.canvas.draw()


if __name__ == '__main__':
  global fig, ax1, ax2
  plt.ion()
  fig = plt.figure(figsize=(8, 8), facecolor='darkgray', edgecolor='black')
  ax1 = fig.add_subplot(211, facecolor='black')
  ax2 = fig.add_subplot(212, facecolor='black')
  ax1.autoscale(True)
  ax2.autoscale(True)
  plt.title('profile')
  plt.ylabel('metrics')
  plt.xlabel('steps')
  fig.show()
  parser = argparse.ArgumentParser()
  parser.add_argument('--flow', default=False, type=bool)
  parser.add_argument('--trim_level', default=None, type=float)
  parser.add_argument('--stage_size', default=1000, type=int)
  parser.add_argument('--window_size', default=100, type=int)
  parser.add_argument('--emp', default=10, type=int)
  args = parser.parse_args()

  try:
    pipe_loss(args.flow, args.trim_level, args.stage_size, args.emp, args.window_size)
    time.sleep(10000)
  except KeyboardInterrupt:
    pass

