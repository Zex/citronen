import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def show_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(losses))+1, losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss/Iteration')
    return fig, ax

def get_loss(line):
    #Iteration 83, loss = 0.08709494
    iteration, loss = re.search('\d+', line), re.search('[0-9]*\.[0-9]+', line)
    if loss is None:
        loss = line.strip('\n')
    else:
        iteration, loss = iteration.group(), loss.group()
        print('epoch:{}, loss:{}'.format(iteration, loss))
    return float(loss)

def pipe_stdin():
    losses = []
    while True:
        line = sys.stdin.readline()
        if line is None or len(line) == 0:
            break
        losses.append(get_loss(line))
    show_losses(losses)
    plt.show()

pipe_stdin()

