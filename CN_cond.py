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

ch_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
                'tau_m': 3.8,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 0.94/0.37,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -41.7
               }

on_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               'tau_m': 2.9,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 4.88/0.37,#2.5,#
               'e_rev_E': -25.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }

dB = 40#20
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
an_spikes = cochlea_file['scaled_times']

w2s_target = 1.#1.5#4.5#0.12#2.5#5.
number_of_inputs = len(an_spikes)#
input_spikes = an_spikes

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,32)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes))
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
ch_pop = sim.Population(number_of_inputs,sim.IF_cond_exp,ch_params_cond,label="ch_fixed_weight_scale_cond")
on_ch_ratio = 0.1
on_pop_size = int(number_of_inputs*on_ch_ratio)
on_pop = sim.Population(on_pop_size,sim.IF_cond_exp,on_params_cond,label="on_fixed_weight_scale_cond")

# cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =

ch_pop.record(["spikes","v"])
on_pop.record(["spikes","v"])

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
an2ch_list = normal_dist_connection_builder(number_of_inputs,number_of_inputs,RandomDistribution,NumpyRNG(),conn_num,dist,sigma)
an_ch_projection = sim.Projection(input_pop,ch_pop,sim.FromListConnector(an2ch_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))

#TODO: calculate formulas for these variables so we can scale
n_on_inputs = 100.#50.
# an_on_p_connect = n_on_inputs/number_of_inputs
an_on_p_connect = 0.45
# an_on_weight = w2s_target/(n_on_inputs)
# an_on_weight = 1.5/(n_on_inputs)
av_weight = 0.0015#0.0055#0.005#0.0025
# an_on_weight =RandomDistribution('uniform',[0.9*av_weight,1.1*av_weight])
# an_on_weight =RandomDistribution('uniform',[0,av_weight])
an_on_weight =RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,2*av_weight])

an_on_projection = sim.Projection(input_pop,on_pop,sim.FixedProbabilityConnector(p_connect=an_on_p_connect),sim.StaticSynapse(weight=an_on_weight))

sigma = 2.
conn_num = int(sigma / in2out_sparse)
on2on_weight = RandomDistribution('uniform',[0.,1.])
# on2on_list = normal_dist_connection_builder(on_pop_size,on_pop_size,RandomDistribution,NumpyRNG(),conn_num,dist,sigma)
# on_on_projection = sim.Projection(on_pop,on_pop,sim.FromListConnector(on2on_list),synapse_type=sim.StaticSynapse(weight=on2on_weight),receptor_type='inhibitory')
on_on_projection = sim.Projection(on_pop,on_pop,sim.FixedProbabilityConnector(p_connect=0.01),
                                  synapse_type=sim.StaticSynapse(weight=on2on_weight),receptor_type='inhibitory')

on_ch_projection = sim.Projection(on_pop,ch_pop,sim.FixedProbabilityConnector(p_connect=0.1),
                                  synapse_type=sim.StaticSynapse(weight=on2on_weight),receptor_type='inhibitory')

duration = 1000.#max(input_spikes[0])

sim.run(duration)
ch_data = ch_pop.get_data(["spikes","v"])
on_data = on_pop.get_data(["spikes","v"])
sim.end()

ch_mem_v = ch_data.segments[0].filter(name='v')
cell_voltage_plot_8(ch_mem_v, plt, duration, [],id=599,scale_factor=0.001,title='ch pop')
on_mem_v = on_data.segments[0].filter(name='v')
cell_voltage_plot_8(on_mem_v, plt, duration, [],id=59,scale_factor=0.001,title='on pop')

ch_spikes = ch_data.segments[0].spiketrains
on_spikes = on_data.segments[0].spiketrains

spike_raster_plot_8(ch_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="ch pop activity")
spike_raster_plot_8(on_spikes,plt,duration/1000.,on_pop_size+1,0.001,title="on pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")

psth_plot_8(plt,numpy.arange(550,650),an_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_AN")
psth_plot_8(plt,numpy.arange(550,650),ch_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_CH")
psth_plot_8(plt,numpy.arange(55,65),on_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_ON")

plt.show()