# Plot from tensorflow event data
# Author: Zex Li <top_zlynch@yahoo.com>

from tensorflow.python.summary import event_accumulator
import matplotlib
matplotlib.use('TkAgg')
import pylab
import numpy as np

marks = ['o', '-', '^', '+', 'x', ':', '.', '--']
colors = ['b', 'g', 'r', 'lightblue']
mfcs = ['pink', 'm', 'c', 'k']
#np.random.shuffle(mfcs)

def plot_from_tfevent(tfevent_base):
    evacc = event_accumulator.EventAccumulator(tfevent_base)
    evacc.Reload()
    print("path:{} file version:{}".format(tfevent_base, evacc.file_version))
    total = len(evacc._scalars.Keys())

    for k in evacc._scalars.Keys():
        metric = evacc._scalars.Items(k)
        pylab.plot(np.arange(len(metric)), [l.value for l in metric], 
                marks[np.random.randint(total)],
                color=colors[np.random.randint(total)],
                mfc=mfcs[np.random.randint(total)])
    pylab.legend(evacc._scalars.Keys(), loc='upper right')
    pylab.xlabel('epoch')
    pylab.ylabel('metrics')
    pylab.title('TF Event Metrics')
    pylab.show()

if __name__ == '__main__':
    tfevent_base = '../build/quora.log'
    plot_from_tfevent(tfevent_base)
    
