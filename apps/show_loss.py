import numpy as np
import matplotlib.pyplot as plt
import pylab
import re
import sys

def show_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(losses)), losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss/Iteration')
    return fig, ax

def show_accuracy(accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(accuracy)), accuracy)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    return fig, ax

def get_accu(line):
    # score: 0.08709494
    accu = re.search('[0-9]*\.[0-9]+', line)
    accu = accu.group()
    print('accuracy:{}'.format(accu))
    return float(accu)

def get_loss(line):
    #Iteration 83, loss = 0.08709494
    iteration, loss = re.search('\d+', line), re.search('[0-9]*\.[0-9]+', line)
    if iteration is not None and loss is not None:
        iteration, loss = iteration.group(), loss.group()
        print('iteration:{}, loss:{}'.format(iteration, loss))
    elif loss is None:
        loss = line.strip('\n')
        if len(loss) == 0 or len(loss.split()) > 0:
            loss = 1.0
        print('iteration:{}, loss:{}'.format('-', loss))
    else:
        loss = 1.0
    return float(loss)

def pipe_stdin_loss():
    losses = []
    while True:
        line = sys.stdin.readline()
        if line is None or len(line) == 0:
            break
        losses.append(get_loss(line))
    show_losses(losses)
    plt.show()

def pipe_stdin_accu():
    accuracy = []
    while True:
        line = sys.stdin.readline()
        if line is None or len(line) == 0:
            break
        accuracy.append(get_accu(line))
    show_accuracy(accuracy)
    plt.show()

pipe_stdin_loss()
#pipe_stdin_accu()

