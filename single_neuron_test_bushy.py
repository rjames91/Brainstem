import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pyNN.utility.plotting import Figure, Panel
from elephant.statistics import isi,cv

#================================================================================================
# Simulation parameters
#================================================================================================
bushy_params_cond = {#'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 1.,#10.0,#2.,#3.,#
               'tau_syn_E': 0.15,#2.5,#
               #'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -65.,#-70.0,
               'v_rest': -65.,
               }

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
results_file = "/cn_tone_1000Hz_stereo_1s_1000an_fibres_0.1ms_timestep_100dB_1s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_0dB_0s_moc_True_lat_True.npz"
results_data = np.load(results_directory+results_file)
an_spikes = results_data['an_spikes'][0]

# cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_1000fibres.npz'.format(dB))
# an_spikes = [[i*7.+ 3.*(np.random.rand()-0.5) for i in range(50)]for _ in range(100)]#[[10.,11.,12.,13.]]#cochlea_file['scaled_times']
# an_spikes = [[10.,15.,20.,100.,105.]]#,102,104]]
# spike_times = [10.,15.,20.,100.,105.]
# spike_times = [50.,105.]
test_dur_ms = 200#
spike_times = [i for i in range(1,test_dur_ms,10)]
# an_spikes = []#,102,104]]

#plt.hist(connection_weight.next(1000),bins=100)
#plt.show()

number_of_inputs = len(an_spikes)#
n_total = int(2.4 * number_of_inputs)
n_b = int(n_total * 55./89)#number_of_inputs#
n_o = int(n_total * 10./89.)
pop_size = max([number_of_inputs,n_b])
# inh_weight = initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

input_spikes = an_spikes
# input_spikes =[]
inh_spikes = []

#================================================================================================
# SpiNNaker setup
#================================================================================================
timestep = 0.1
sim.setup(timestep=timestep)
# sim.setup(timestep=1.)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
input_pops=[]
# input_spikes = [10.]
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")

# cd_pop = sim.Population(n_b,sim.extra_models.Izhikevich_cond,bushy_params_cond,label="fixed_weight_scale_cond_1")
# cd_pop_2 = sim.Population(n_b,sim.extra_models.Izhikevich_cond,bushy_params_cond,label="fixed_weight_scale_cond_2")
cd_pop = sim.Population(n_b,sim.IF_cond_exp,bushy_params_cond,label="fixed_weight_scale_cond_1")

cd_pop.record("all")
#================================================================================================
# Projections
#================================================================================================
n_an_b_connections = RandomDistribution('uniform',[2.,5.])
w2s_b = 5.#
av_an_b = w2s_b/5.
an_b_weight = RandomDistribution('normal_clipped',[av_an_b,0.1*av_an_b,0,av_an_b*2.])
an_b_list, max_dist = normal_dist_connection_builder(number_of_inputs, n_b, RandomDistribution,
                                                     conn_num=n_an_b_connections, dist=1.,
                                                     sigma=1., conn_weight=an_b_weight, delay=timestep,
                                                     normalised_space=pop_size, get_max_dist=True)

# sim.Projection(input_pop,cd_pop,sim.FromListConnector([(0,0)]),synapse_type=sim.StaticSynapse(weight=w2s_b))
sim.Projection(input_pop, cd_pop, sim.FromListConnector(an_b_list),
                                       synapse_type=sim.StaticSynapse())

# sim.Projection(input_pop, cd_pop_2, sim.FromListConnector(an_b_list),
#                                        synapse_type=sim.StaticSynapse())


duration = test_dur_ms#max(input_spikes[0])

sim.run(duration)

cd_data =[]
cd_data.append(cd_pop.get_data())
# input_data = input_pop.get_data()

sim.end()

for i in range(1):
    title = "LIF neuron_{}".format(i)
    plt.figure(title)
    spike_raster_plot_8(cd_data[i].segments[0].spiketrains,plt,duration/1000.,n_b+1,0.001,title=title,subplots=(4,1,1),markersize=1,)
    mem_v = cd_data[i].segments[0].filter(name='v')
    cell_voltage_plot_8(mem_v, plt, duration, [],id=n_b/2,scale_factor=0.0001,title="",subplots=(4,1,2))
    plt.ylabel("membrane voltage (mV)")
    gsyn = cd_data[i].segments[0].filter(name='gsyn_exc')
    cell_voltage_plot_8(gsyn, plt, duration, [],scale_factor=0.0001,title="",subplots=(4,1,3))
    spike_raster_plot_8(an_spikes, plt, duration / 1000., number_of_inputs + 1, 0.001,subplots=(4, 1, 4),markersize=1,)

plt.show()