import numpy as np
from scipy.io import wavfile,loadmat
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv


plot_spikes = False
plot_moc = False
plot_stim =False
plot_debug = True
# Open the results
results_file = "./debug.npz"
n_ears = 2

# duration = binaural_audio_data[0].size/Fs
duration = 0.4#2.45#0.25#0.75#1.#
results_data = np.load(results_file)
Fs = results_data['Fs']
an_spikes = results_data['ear_spikes']
moc_att = results_data['moc_data']
stimulus = results_data['stimulus']
ear_debug = results_data['ear_debug']


for ear_index in range(n_ears):
    neuron_title_list = ['an']
    neuron_list = []
    for i, neuron_times in enumerate(neuron_list):
        non_zero_neuron_times = neuron_times[ear_index]

        if plot_spikes:
            plt.figure("spikes ear{}".format(ear_index))
            spike_raster_plot_8(non_zero_neuron_times, plt, duration, len(non_zero_neuron_times) + 1, 0.001,
                                title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1))

    if plot_moc:
        plt.figure("moc ear {}".format(ear_index))
        n_channels = len(moc_att[ear_index])
        # for i,moc in enumerate(moc_att[ear_index][::n_channels/10]):
        for i,moc in enumerate([moc_att[ear_index][0]]):
            if moc.min()==1:
                print "ear {} channel {}".format(ear_index,i)
            if 1:#moc.min()<0.9:
                t = np.linspace(0,duration*1000.,len(moc))
                plt.plot(t,moc)
        plt.ylabel("MOC attenuation")
        plt.xlabel("time (ms)")

    if plot_stim:
        plt.figure("audio stimulus")
        ax = plt.subplot(len(stimulus), 1, ear_index+1)
        ax.set_title('ear {}'.format(ear_index))
        x = np.linspace(0,duration,len(stimulus[ear_index]))
        ax.plot(x,stimulus[ear_index])

    if plot_debug:
        plt.figure("debug hsr ear{}".format(ear_index))
        for moc_signal in ear_debug[ear_index][0::2]:
            x = np.linspace(0, duration, len(moc_signal))
            plt.plot(x, moc_signal)
        plt.xlabel("time (ms)")
        plt.figure("debug lsr ear{}".format(ear_index))
        for moc_signal in ear_debug[ear_index][1::2]:
            x = np.linspace(0, duration, len(moc_signal))
            plt.plot(x, moc_signal)
        plt.xlabel("time (ms)")


an_count = 0
for ear in an_spikes:
    for neuron in ear:
        an_count+=len(neuron)
print "an spike count = {}".format(an_count)

plt.show()