from scipy.io import loadmat
from signal_prep import *
import matplotlib.pylab as plt
import numpy as np

file_name = '/home/rjames/Dropbox (The University of Manchester)/EarProject/MAP_BS_17/results/n_channels_experiment/30chan_30k.mat'
#'./a1_spikes.mat'#'./a1_spikes_pre.mat'#
matlab_spikes = loadmat(file_name)
spike_times = []
neurons = matlab_spikes['ANoutput']
an_size = len(neurons)
max_time = 0
for neuron in neurons:
    times = np.nonzero(neuron)[0]*(1000./50e3)
    spike_times.append(times)
    if len(times)>0:
        if times.max()>max_time:
            max_time=times.max()

split_times = [spike_times[:an_size/2],spike_times[an_size/2:]]
interleaved_times = [val for tup in zip(*split_times) for val in tup]

# spike_raster_plot_8(interleaved_times, plt, max_time / 1000., an_size + 1, 0.001,ylims=(15000,20000),xlim=(0.2,0.4),markersize=1)
spike_raster_plot_8(interleaved_times, plt, max_time / 1000., an_size + 1, 0.001,xlim=(0.21,0.41),markersize=1)#,ylims=(15000,20000))

# ids = [id for id in matlab_spikes['output'][0]]
# max_id = max(ids)
# times = [time*1000. for time in matlab_spikes['output'][1]]
# max_time = max(times)
# spike_train = [(ids[i],times[i]) for i in range(len(ids))]
# #numpy.save("./spike_times_an_alt",[spike_train,max_time/1000.])
#
# spike_raster_plot(spike_train,plt=plt,duration=max_time/1000.,ylim=max_id,scale_factor=0.001,title=file_name+" spikes")
#
# psth_plot(plt,numpy.arange(400,600),spike_train,bin_width=0.001,duration=max_time/1000.,scale_factor=0.001,title=file_name+" PSTH")

#timestep =1.
#spike_times_spinn=[]
#for i in range(1000):
#        id_times = [1000*timestep*spike_time for (neuron_id,spike_time) in spike_train if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn_an_alt",spike_times_spinn)

plt.show()

print


