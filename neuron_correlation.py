import numpy as np
from signal_prep import *
import matplotlib.pylab as plt
from scipy.io import loadmat

#spike_train = [(0,30),(0,50),(2,35),(10,100),(10,101)]
spike_train = np.load('./ac_spikes.npy')#np.load('./belt_spikes.npy')#

matlab_spikes = loadmat('./belt_spikes.mat')
ids = [id for id in matlab_spikes['output'][0]]
times = [time*1000. for time in matlab_spikes['output'][1]]
max_time = max(times)
#spike_train = [(ids[i],times[i]) for i in range(len(ids))]

#ids = [id for (id,time) in spike_train]
max_id = len(spike_train)#max(ids)
num_classes = 2
timestep =1.
time_window = 600

#open an spikes to identify stimulus times
# [spikes_train_an_ms,duration] = np.load('./spike_times_an_alt.npy')#
#[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_1sp_1num_1000rep.npy")
[spike_trains,an_scale_factor]=numpy.load("../OME_SpiNN/spike_trains_1sp_1num_1000rep.npy")

duration = 420.
print "duration= {} seconds".format(duration)
#an_scale_factor = 1./Fs#duration/numpy.max(spike_times)
spikes_train_an_ms = [(neuron_id,int(1000*timestep*an_scale_factor*spike_time)) for (neuron_id,spike_time) in spike_trains if (spike_time*an_scale_factor)<=duration]

#psth_plot(plt,numpy.arange(1000),spike_train,bin_width=0.01,duration=duration,scale_factor=0.001,title="PSTH_Belt")
stimulus_times = stimulus_onset_detector(spikes_train_an_ms,1000,duration,num_classes)

#spike_raster_plot(spike_train,plt=plt,duration=duration,ylim=1000 ,scale_factor=0.001,title='A1')
#spike_raster_plot(spikes_train_an_ms,plt=plt,duration=duration,ylim=1000 ,scale_factor=0.001,title='AN')
#psth_plot(plt,numpy.arange(1000),spike_train,bin_width=0.001,duration=duration,scale_factor=0.001,title="PSTH_A1_pre")
#plt.show()

#only interested in responses to final few stimuli to observe plasticity effects
stimulus_times_final = []
for stimulus_time in stimulus_times:
    stimulus_times_final.append(stimulus_time[-10:])

counts,selective_neuron_ids,significant_spike_count = neuron_correlation(spike_train,time_window,stimulus_times_final,max_id)
print "significant spike count: {}".format(significant_spike_count)
import matplotlib.pyplot as plt
max_count = counts.max()
plt.figure()
title = "{}ms post-stimulus spike count for AC A1 layer".format(time_window)
plt.title(title)
plt.xlabel("neuron ID")
plt.ylabel("spike count")
plt.plot(counts.T)
plt.legend(["stimulus 'one'", "stimulus 'two'"])
plt.ylim((0,max_count+1))

#plt.figure()
#plt.hist(counts[0])
#plt.figure()
#plt.hist(counts[1])

np.save('./selective_ids.npy',selective_neuron_ids)
for i in range(len(selective_neuron_ids)):
    print selective_neuron_ids[i]

plt.show()
