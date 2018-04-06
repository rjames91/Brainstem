import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import numpy as np

sim_duration = 10000.#46349.#
w_max = 0.25#10.

varying_weights = numpy.load("./weights_to_belt.npy")
ids = RandomDistribution('uniform',(0,99))
chosen=ids.next(n=12)
chosen_int = [int(id) for id in chosen]
#chosen_int = [380]#single target id

vary_weight_plot(varying_weights,chosen_int,[],sim_duration,
                         plt,np=numpy,num_recs=int(np.ceil(sim_duration/4000)),ylim=w_max+(w_max/10.))

weight_dist_plot(varying_weights,1,plt)

#[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_10sp_2num_5rep.npy")
#spikes_train_an_ms = [(neuron_id,int(1000*(1./22050.)*spike_time)) for (neuron_id,spike_time) in spike_trains]
#psth_plot(plt, numpy.arange(1000), spikes_train_an_ms, bin_width=0.001, duration=sim_duration / 1000.,
#         scale_factor=0.001, title="PSTH_AN")

plt.show()