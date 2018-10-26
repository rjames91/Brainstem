import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
from pyNN.utility.plotting import Figure, Panel


#================================================================================================
# Simulation parameters
#================================================================================================
model = sim.IF_curr_exp
target_cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 1.0,#2.5,
               'tau_syn_I': 1.0,#2.5,
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
               'e_rev_E': -15.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }
on_params = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               'tau_m': 2.9,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 4.88/0.37,#2.5,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }

octopus_params_cond = {'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -20.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.6,#-70.0,
               'v_rest': -60.6,
               'v_thresh': -56.
               }

t_stellate_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
                'tau_m': 3.8,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 0.94,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -41.7
               }

d_stellate_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               'tau_m': 2.9,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 4.88,#2.5,#
               'e_rev_E': -30.,#-35.,#-25.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }

one_to_one_cond_params = {
                'tau_m': .1,
                'tau_syn_E': .1,
                'v_thresh': -64.
}

dB = 50#20
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_1000fibres.npz'.format(dB))
# an_spikes = [[i*7.+ 3.*(np.random.rand()-0.5) for i in range(50)]for _ in range(100)]#[[10.,11.,12.,13.]]#cochlea_file['scaled_times']
an_spikes = [[10.]]#,102,104]]

# an_spikes = []
# for _ in range(60):
#     an_spikes.append([10. + (5. * (np.random.rand()-0.5))])
# an_spikes = cochlea_file['scaled_times']
target_pop_size =1
w2s_target = 2.#7.#0.7#1.#15.#0.005#0.0015#0.0006#1.5#4.5#0.12#2.5#5.
# n_connections = 120.#50#100
n_connections = RandomDistribution('uniform',[30.,120.])
# connection_weight = w2s_target/n_connections#w2s_target#initial_weight*2.#/2.
av_weight =w2s_target#/30.#w2s_target/90.# w2s_target/n_connections#

#plt.hist(connection_weight.next(1000),bins=100)
#plt.show()

number_of_inputs = len(an_spikes)#
# inh_weight = initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

input_spikes = an_spikes
# input_spikes =[]
inh_spikes = []
isi = 3.
n_repeats = 20

# for neuron in range(number_of_inputs):
#      input_spikes.append([i*isi for i in range(n_repeats) if i<50 or i>100])
#     inh_spikes.append([(i*isi)-1 for i in range(n_repeats) if i<5 or i>10])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params,label="fixed_weight_scale")
# cd_pop = sim.Population(target_pop_size,sim.IF_curr_exp,one_to_one_cond_params,label="fixed_weight_scale")
cd_pop = sim.Population(target_pop_size,sim.IF_curr_exp,one_to_one_cond_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,on_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =
input_pop.record("spikes")
cd_pop.record("all")
#================================================================================================
# Projections
#================================================================================================
# diagonal_width = 2.#26.#
# diagonal_sparseness = 1.
# in2out_sparse = .67 * .67 / diagonal_sparseness
# dist = max(int(number_of_inputs / number_of_inputs), 1)
# sigma = dist * diagonal_width
# conn_num = int(sigma / in2out_sparse)
# av_weight = w2s_target/conn_num
# # an2ch_weight = RandomDistribution('normal',[av_weight,0.1])
# # an2ch_weight = RandomDistribution('normal',[av_weight,0.5])
# # an2ch_weight = RandomDistribution('uniform',[0,av_weight+(av_weight*1.5)])
# an2ch_weight = RandomDistribution('uniform',[0.9*(w2s_target/conn_num),1.1*(w2s_target/conn_num)])
#
# an2ch_list = normal_dist_connection_builder(number_of_inputs,number_of_inputs,RandomDistribution,
#                                             conn_num,dist,sigma)

# connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,2*av_weight])
# connection_weight = RandomDistribution('uniform',[0,av_weight*2.])
connection_weight = av_weight#w2s_target/number_of_inputs
# an_on_list = normal_dist_connection_builder(number_of_inputs,target_pop_size,RandomDistribution,conn_num=n_connections,dist=1.,sigma=number_of_inputs/6.
#                                             ,conn_weight=connection_weight)

# input_projection = sim.Projection(input_pop,cd_pop,sim.FromListConnector(an_on_list),synapse_type=sim.StaticSynapse())
# input_projection = sim.Projection(input_pop,cd_pop,sim.FromListConnector(an2ch_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))
input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
# input_projection = sim.Projection(input_pop,cd_pop,sim.FixedProbabilityConnector(p_connect=n_connections/number_of_inputs),synapse_type=sim.StaticSynapse(weight=connection_weight))
#inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
# inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=stdp_model_cd,receptor_type='inhibitory')

duration = 30.#max(input_spikes[0])

sim.run(duration)

cd_data = cd_pop.get_data()
input_data = input_pop.get_data()

sim.end()

Figure(
    # plot data for postsynaptic neuron
    Panel(cd_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",legend=False,
          yticks=True,xticks=True, xlim=(0, duration)),
    Panel(cd_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",legend=False,
           yticks=True,xticks=True, xlim=(0, duration)),
    # Panel(cd_data.segments[0].filter(name='gsyn_inh')[0],
    #       ylabel="gsyn inhibitory (mV)",legend=False,
    #        yticks=True,xticks=True, xlim=(0, duration)),
    Panel(cd_data.segments[0].spiketrains,marker='.',
          yticks=True,markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0,xticks=True, xlim=(0, duration)),
    Panel(input_data.segments[0].spiketrains, marker='.',
          yticks=True, markersize=3,
          markerfacecolor='black', markeredgecolor='none',
          markeredgewidth=0, xticks=True, xlim=(0, duration))
)

# mem_v = cd_data.segments[0].filter(name='v')
# # cell_voltage_plot_8(mem_v, plt, duration, [],id=599,scale_factor=0.001,title='cd pop')
#
# ch_spikes = cd_data.segments[0].spiketrains
#
# spike_raster_plot_8(ch_spikes,plt,duration/1000.,target_pop_size+1,0.001,title="cd pop activity")
# spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")
# cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.0001,title='cd pop')

# psth_plot_8(plt,numpy.arange(550,650),an_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_AN")
# psth_plot_8(plt,numpy.arange(target_pop_size),ch_spikes,bin_width=0.001,duration=duration/1000.,title="PSTH_CH")

plt.show()