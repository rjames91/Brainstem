import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

#================================================================================================
# Simulation parameters
#================================================================================================
model = sim.IF_curr_exp
target_cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               'tau_refrac': 1.0,#2.0,#
              # 'tau_syn_E': 1.0,#2.5,
              # 'tau_syn_I': 1.0,#2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

inh_cond_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 10.0,#2.5,
               #'tau_syn_I': 1.0,#2.5,
               'e_rev_E': -50.0,
               'e_rev_I': 0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh':-45.0#-55.4#
               }

neuron_params = {
    "v_thresh": 100,
    "v_reset": 0,
    "v_rest": 0,
    "i_offset": 0,
    "e_rev_E": 80,
#     "tau_syn_E":50,
    "e_rev_I": 0 # DC input
                 }

ex_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               # 'tau_m': 5.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
               'e_rev_E': -49.,#-55.1,#
               'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh': -50.0
               }

dB = 40#20
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
an_spikes = cochlea_file['scaled_times']

w2s_target = 1.#1.5#4.5#0.12#2.5#5.
n_connections = 1
initial_weight = w2s_target/n_connections
connection_weight = w2s_target#initial_weight*2.#/2.
number_of_inputs = len(an_spikes)#
inh_weight = initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

input_spikes = an_spikes
# input_spikes =[]
inh_spikes = []
isi = 3.
n_repeats = 150

# for neuron in range(number_of_inputs):
#      input_spikes.append([i*isi for i in range(n_repeats) if i<50 or i>100])
#     inh_spikes.append([(i*isi)-1 for i in range(n_repeats) if i<5 or i>10])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes))
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
cd_pop = sim.Population(number_of_inputs,sim.IF_cond_exp,ex_params_cond,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =

cd_pop.record(["spikes","v"])

#================================================================================================
# Projections
#================================================================================================
diagonal_width = 2.#26.#
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(number_of_inputs / number_of_inputs), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
av_weight = w2s_target/conn_num
# an2ch_weight = RandomDistribution('normal',[av_weight,0.1])
# an2ch_weight = RandomDistribution('normal',[av_weight,0.5])
# an2ch_weight = RandomDistribution('uniform',[0,av_weight+(av_weight*1.5)])
an2ch_weight = RandomDistribution('uniform',[0.9*(w2s_target/conn_num),1.1*(w2s_target/conn_num)])

an2ch_list = normal_dist_connection_builder(number_of_inputs,number_of_inputs,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma)

input_projection = sim.Projection(input_pop,cd_pop,sim.FromListConnector(an2ch_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))
# input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
#inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
# inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=stdp_model_cd,receptor_type='inhibitory')

duration = 1000.#max(input_spikes[0])

sim.run(duration)

cd_data = cd_pop.get_data(["spikes","v"])

sim.end()

mem_v = cd_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, [],id=599,scale_factor=0.001,title='cd pop')

ch_spikes = cd_data.segments[0].spiketrains

spike_raster_plot_8(ch_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="cd pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")

psth_plot_8(plt,numpy.arange(550,650),an_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_AN")
psth_plot_8(plt,numpy.arange(550,650),ch_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_CH")

plt.show()