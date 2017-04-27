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
    accu = re.search('score:[0-9]*\.[0-9]+', line)
    if accu is None:
        return None
    accu = accu.group().split(':')[1]
    print('accuracy:{}'.format(accu))
    return float(accu)

def get_loss(line):
    #Iteration 83, loss = 0.08709494
    line = line.strip('\n')
    iteration, loss = re.search('\d+', line), re.search('loss = [0-9]*\.[0-9]+', line)
    if iteration is not None and loss is not None:
        iteration, loss = iteration.group(), loss.group()
        loss = loss.split(' ')[2]
        if len(loss) == 0:
            return None
        print('iteration:{}, loss:{}'.format(iteration, loss))
    else:
        return None
    return float(loss)

def pipe_stdin_loss():
    losses = []
    while True:
        line = sys.stdin.readline()
        if line is None or len(line) == 0:
            break
        loss = get_loss(line)
        if loss is not None:
            losses.append(loss)
    show_losses(losses)
    plt.show()

def pipe_stdin_accu():
    accuracy = []
    while True:
        line = sys.stdin.readline()
        print(line)
        if line is None or len(line) == 0:
            break
        accu = get_accu(line)
        if accu is not None:
            accuracy.append(accu)
    show_accuracy(accuracy)
    plt.show()

pipe_stdin_loss()
#pipe_stdin_accu()

