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
octopus_params_cond = {'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.6,#-70.0,
               'v_rest': -60.6,
               'v_thresh': -56.
               }

dB = 50#20
duration = 300.#75000.#20000.#

input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_10000fibres.npz'.format(dB))
cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_5000fibres.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_aiu_60s_{}dB_1000fibres.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_20s_50dB_1000fibres.npz'.format(dB))
# an_spikes = [[100.,102,104]]
an_spikes = cochlea_file['scaled_times']
# input_spikes = cochlea_file['scaled_times']

input_spikes =[]
for neuron in an_spikes:
    input_spikes.append([spike_time for spike_time in neuron if spike_time <= duration])

# an_size = len(input_spikes)
# max_psth = 0
# for _ in range(100):
#     chosen_ids = np.random.choice(np.arange(int(0.9*an_size),an_size),size=100,replace=False)
#     psth = psth_plot_8(plt,chosen_ids,input_spikes,bin_width=0.001,duration=duration/1000.,title="PSTH_AN_{}".format(an_size))
#     if psth.max() > max_psth:
#         max_psth = psth.max()
# plt.ylim((0,max_psth))
# spike_raster_plot_8(input_spikes,plt,duration/1000.,an_size+1,0.001,title="input activity_{}".format(an_size))
# print "max psth:{} (sp/s)".format(max_psth)
# plt.show()

onset_times = cochlea_file['onset_times']
w2s_target = 7.#15.#0.005#0.0015#0.0006#1.5#4.5#0.12#2.5#5.
# n_connections = RandomDistribution('uniform',[30.,120.])
n_connections = RandomDistribution('uniform',[30.,120.])
# n_connections = RandomDistribution('uniform',[30.,60.])
# av_weight =(w2s_target/30.)#w2s_target/90.# w2s_target/n_connections#
av_weight =(w2s_target/45.)#w2s_target/90.# w2s_target/n_connections#

#plt.hist(connection_weight.next(1000),bins=100)
#plt.show()

number_of_inputs = len(input_spikes)
# input_spikes = an_spikes

n_total = 2. * number_of_inputs
target_pop_size = int(np.round(n_total*10./89.)) #200

# variation = []
# n_conn = []
# max_psth = []
# for _ in range(500):
#     an_on_list = normal_dist_connection_builder(number_of_inputs, target_pop_size, RandomDistribution,
#                                                 conn_num=n_connections, dist=1., sigma=number_of_inputs / 20.
#                                                 # , conn_weight=1.,posts=[800],multapses=True)
#                                                 , conn_weight=1.,posts=[403],multapses=True)
#                                                 # , conn_weight=1.,posts=[96],multapses=True)
#     test_list = np.asarray([source for (source,target,weight,delay) in an_on_list])
#     chosen_ids = test_list
#     psth = psth_plot_8(plt,chosen_ids,input_spikes,bin_width=0.001,duration=duration/1000.,title="PSTH_AN_{}".format(number_of_inputs))
#     max_psth.append(psth.max())
#     variation.append(test_list.max()-test_list.min())
#     n_conn.append(test_list.size)
#
# plt.ylim((0,max(max_psth)))
# spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity_{}".format(number_of_inputs))
# print '{} fibres average max psth:{} (sp/s) average pre neuron ID variation:{} average number of incoming connections per post:{}'.format(number_of_inputs,np.mean(max_psth),np.mean(variation),np.mean(n_conn))
#
# # plt.figure()
# plt.show()

#================================================================================================
# SpiNNaker setup
#================================================================================================
timestep = 1.#0.1
sim.setup(timestep=timestep)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,16)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,32)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
# input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")
# oct_pop = sim.Population(target_pop_size,sim.IF_cond_exp,octopus_params_cond,label="fixed_weight_scale_cond")
# oct_pop.record(["spikes","v"])
# oct_pop.record(["spikes"])

#================================================================================================
# Projections
#================================================================================================
connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/10.,0,av_weight*2.])
an_on_list = normal_dist_connection_builder(number_of_inputs,target_pop_size,RandomDistribution,conn_num=n_connections,dist=1.,sigma=number_of_inputs/20.
                                            ,conn_weight=connection_weight)

# input_projection = sim.Projection(input_pop,oct_pop,sim.FromListConnector(an_on_list),synapse_type=sim.StaticSynapse())

input_pops,target_pops,an_on_projs,m_pre = sub_pop_builder_inter(sim,target_pop_size,sim.IF_cond_exp,octopus_params_cond,"SSA",input_spikes,
                                                            "an_input","octopus",an_on_list)

max_period = 5000.
num_recordings =int((duration/max_period)+1)

# sim.SpiNNaker.transceiver.set_watch_dog(False)
for i in range(num_recordings):
    sim.run(duration/num_recordings)

# octopus_data = oct_pop.get_data(["spikes","v"])
# octopus_data = oct_pop.get_data(["spikes"])
octopus_spikes = get_sub_pop_spikes(target_pops)

sim.end()

# octopus_spikes = octopus_data.segments[0].spiketrains

spike_raster_plot_8(octopus_spikes,plt,duration/1000.,target_pop_size+1,0.001,title="octopus pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")

if 0:#duration < 5000.:
    mem_v = octopus_data.segments[0].filter(name='v')
    cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,title='octopus pop')
    psth_plot_8(plt,numpy.arange(target_pop_size),octopus_spikes,bin_width=0.001,duration=duration/1000.,title="PSTH_OCT")

# np.savez_compressed(input_directory+'/octopus_13.5_1kHz_{}dB_{}ms_timestep_{}s'.format(dB,timestep,int(duration/1000.)),an_spikes=an_spikes,
#                     octopus_spikes=octopus_spikes,onset_times=onset_times)

# psth_plot_8(plt,np.arange((9000,10000)),input_spikes,bin_width=0.001,duration=2.,title="PSTH_AN")

plt.show()