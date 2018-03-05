from scipy.io import loadmat
from signal_prep import *
import matplotlib.pylab as plt

file_name = './a1_spikes.mat'#'./a1_spikes_pre.mat'#
matlab_spikes = loadmat(file_name)
ids = [id for id in matlab_spikes['output'][0]]
max_id = max(ids)
times = [time*1000. for time in matlab_spikes['output'][1]]
max_time = max(times)
spike_train = [(ids[i],times[i]) for i in range(len(ids))]
#numpy.save("./spike_times_an_alt",[spike_train,max_time/1000.])

spike_raster_plot(spike_train,plt=plt,duration=max_time/1000.,ylim=max_id,scale_factor=0.001,title=file_name+" spikes")

psth_plot(plt,numpy.arange(400,600),spike_train,bin_width=0.001,duration=max_time/1000.,scale_factor=0.001,title=file_name+" PSTH")

#timestep =1.
#spike_times_spinn=[]
#for i in range(1000):
#        id_times = [1000*timestep*spike_time for (neuron_id,spike_time) in spike_train if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn_an_alt",spike_times_spinn)

plt.show()

print


