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
# octopus_params_cond_izh_orig = {
#                 'a':0.02,
#                 'b':0.25,#0.05,#
#                 'c':-65,
#                 'd':4,
#                 'u':-15,
#                'tau_syn_E': 0.2,#0.35,#2.5,#
#                'e_rev_E': -55.,#-10.,#-10.,#-35.,#-55.1,#
#                'v': -70.,
#                }
octopus_params_cond_izh_orig = {
                'a':0.02,
                'b':0.1,#0.05,#
                'c':-65,
                'd':2,
                'u':-15,
               'tau_syn_E': 0.2,#0.35,#2.5,#
               'e_rev_E': -55.,#-10.,#-10.,#-35.,#-55.1,#
               'v': -70.,
               }

octopus_params_cond_izh = {
                'a':0.02,
                'b':0.1,#0.25,#
                'c':-65,
                'd':2,
                'u':-5,
               'tau_syn_E': 0.2,#0.35,#2.5,#
               'e_rev_E':30.,# -10.,#-35.,#-55.1,#
               'v': -70.,
               # 'i_offset':5.
               }

octopus_params_cond_izh = {
                'a':0.02,
                'b':0.25,#
                'c':-65,
                'd':2,
                'u':-5,
               'tau_syn_E': 0.2,#0.35,#2.5,#
               'e_rev_E':30.,# -10.,#-35.,#-55.1,#
               'v': -70.,
               'i_offset':0.5
               }

octopus_lif_params = {
                'tau_m':0.9,
                'e_rev_E': -40.,
                'v_reset': -70.,
                'v_thresh': -50.,
                'tau_syn_E': 0.2,#1.,#0.5,#
                # 'cm': 10.,
                'tau_refrac':0.
}

izk_B = {
               'a':0.02,#0.2,
               'b':0.25,#0.01,
               'c':-65,
               'd':6,
               'u':-15,
               'tau_syn_E': 1.,#0.2,
               'e_rev_E': -47.,
               #'tau_syn_I': 4.0,#2.5,#
               'v': -70.0,
               # 'i_offset':30.
}
izk_A = {
               'a':0.2,
               'b':0.02,
               'c':-65,#-40,#
               'd':6,
               'u':-15,
               'tau_syn_E': 0.2,
               'e_rev_E': -20.,
               #'tau_syn_I': 4.0,#2.5,#
               'v': -80,#-70.0,#-50.,#
               # 'i_offset':30.
}

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
results_file = "/cn_tone_100Hz_stereo_0s_1000an_fibres_0.1ms_timestep_50dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_100.0Hz_sam_tone1000Hz_0s_1000an_fibres_0.1ms_timestep_50dB_0s_moc_True_lat_True.npz"
results_data = np.load(results_directory+results_file)
sg_spikes = [data.segments[0].spiketrains for data in results_data['sg_data']]
an_spikes=sg_spikes[0]

# cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_1000fibres.npz'.format(dB))
# an_spikes = [[i*7.+ 3.*(np.random.rand()-0.5) for i in range(50)]for _ in range(100)]#[[10.,11.,12.,13.]]#cochlea_file['scaled_times']
# an_spikes = [[10.,15.,20.,100.,105.]]#,102,104]]
# spike_times = [10.,15.,20.,100.,105.]
# spike_times = [50.,105.]
test_dur_ms = 400#
spike_times = [i for i in range(1,test_dur_ms,10)]
# an_spikes = []#,102,104]]
# n_inputs = 1000
# spike_jitter = 0
# for i in range(n_inputs):
#     an_spikes.append([i+spike_jitter*(np.random.rand()-0.5) for i in spike_times])

n_inputs = len(an_spikes)

# an_spikes = []
# for _ in range(60):
#     an_spikes.append([10. + (5. * (np.random.rand()-0.5))])
# an_spikes = cochlea_file['scaled_times']
target_pop_size =1
w2s_target = 0.1# 30.#0.06#1.6#0.3#0.5#0.1#0.2#3.#0.7#1.#15.#0.005#0.0015#0.0006#1.5#4.5#0.12#2.5#5.
# n_connection = 120.#50#100
n_connections = RandomDistribution('uniform',[30.,120.])
# connection_weight = w2s_target/n_connections#w2s_target#initial_weight*2.#/2.
av_weight =w2s_target/n_inputs#/30.#w2s_target/90.# w2s_target/n_connections#

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
# isi = 3.
# n_repeats = 20

# for neuron in range(number_of_inputs):
#      input_spikes.append([i*isi for i in range(n_repeats) if i<50 or i>100])
#     inh_spikes.append([(i*isi)-1 for i in range(n_repeats) if i<5 or i>10])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=0.1)
# sim.setup(timestep=1.)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
input_pops=[]
# input_spikes = [10.]
input_pop = sim.Population(n_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")
# input_pop_2 = sim.Population(n_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")
# input_pop = sim.Population(n_inputs,sim.SpikeSourcePoisson(rate=0.,duration=test_dur_ms*0.4),label="an_pop_input")
# input_pop_2 = sim.Population(n_inputs,sim.SpikeSourcePoisson(rate=500.,duration=test_dur_ms*0.4,start=test_dur_ms*0.6),label="an_pop_input")
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params,label="fixed_weight_scale")
# cd_pop = sim.Population(target_pop_size,sim.IF_curr_exp,one_to_one_cond_params,label="fixed_weight_scale")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,moc_class_2_params,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.IF_cond_exp,moc_lif_params,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(1,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(1,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="fixed_weight_scale_cond")

# t_stellate_izk_class_2_params['d']=100
# t_stellate_izk_class_2_params['i_offset']=50
# octopus_params_cond_izh['i_offset']=50
# octopus_lif_params['i_offset']=50
# cd_pop_2 = sim.Population(1,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="fixed_weight_scale_cond")
cd_pop = sim.Population(n_o,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh_orig,label="fixed_weight_scale_cond_1")
cd_pop_2 = sim.Population(n_o,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="fixed_weight_scale_cond_2")
# cd_pop_2 = sim.Population(n_o,sim.Izhikevich,octopus_params_cond_izh,label="fixed_weight_scale")
# cd_pop_2 = sim.Population(1,sim.IF_cond_exp,octopus_lif_params,label="fixed_weight_scale_cond")
# cd_pop_2 = sim.Population(1,sim.IF_cond_exp,{},label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.IF_cond_exp,{'tau_m':20.,'cm':2.,'i_offset':20.},label="fixed_weight_scale_cond")
# cd_pop_2 = sim.Population(target_pop_size,sim.IF_cond_exp,{'tau_m':20.,'cm':2.,'i_offset':30.},label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,{'v':-65,'d':6,'i_offset':20.},label="fixed_weight_scale_cond")
# cd_pop = sim.Population(1,sim.IF_curr_exp,on_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =
# input_pop.record("spikes")

cd_pop.record("all")
cd_pop_2.record("all")
# cd_pop.record(["v"])
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
# connection_weight = RandomDistribution('normal_clipped',[av_weight,av_weight/100.,0,2*av_weight])
# connection_weight = RandomDistribution('uniform',[av_weight/5.,av_weight*2])
connection_weight = av_weight#w2s_target/number_of_inputs
# an_on_list = normal_dist_connection_builder(number_of_inputs,target_pop_size,RandomDistribution,conn_num=n_connections,dist=1.,sigma=number_of_inputs/6.
#                                             ,conn_weight=connection_weight)
w2s_o = 1.#1.5#3.# 2.#7.
n_an_o_connections = RandomDistribution('uniform', [30., 120.])
# n_an_o_connections = RandomDistribution('normal_clipped',[50.,5.,30.,120.])
av_an_o = w2s_o / 50.
an_o_weight = RandomDistribution('normal_clipped', [av_an_o, 0.1 * av_an_o, 0, av_an_o * 2.])

an_o_list, max_dist = normal_dist_connection_builder(number_of_inputs, n_o, RandomDistribution,
                                                     conn_num=n_an_o_connections, dist=1.,
                                                     sigma=pop_size / 20., conn_weight=an_o_weight,
                                                     normalised_space=pop_size, get_max_dist=True)
sim.Projection(input_pop, cd_pop, sim.FromListConnector(an_o_list),
                                       synapse_type=sim.StaticSynapse())

sim.Projection(input_pop, cd_pop_2, sim.FromListConnector(an_o_list),
                                       synapse_type=sim.StaticSynapse())

# input_projection = sim.Projection(input_pop,cd_pop,sim.FromListConnector(an_on_list),synapse_type=sim.StaticSynapse())
# input_projection = sim.Projection(input_pop,cd_pop,sim.FromListConnector(an2ch_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))
# input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
# input_projection = sim.Projection(input_pop,cd_pop_2,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
# input_projection = sim.Projection(input_pop_2,cd_pop_2,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
# input_projection = sim.Projection(input_pop,cd_pop,sim.FixedProbabilityConnector(p_connect=n_connections/number_of_inputs),synapse_type=sim.StaticSynapse(weight=connection_weight))
#inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
# inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=stdp_model_cd,receptor_type='inhibitory')

# for i, t_mocc_l in enumerate(t_mocc_list):
#     if len(t_mocc_l) > 0:
#         sim.Projection(input_pops[i], cd_pop,sim.FromListConnector(t_mocc_l), synapse_type=sim.StaticSynapse())

duration = test_dur_ms#max(input_spikes[0])

sim.run(duration)

cd_data =[]
cd_data.append(cd_pop.get_data())
cd_data.append(cd_pop_2.get_data())
# input_data = input_pop.get_data()

sim.end()

# Figure(
#     # plot data for postsynaptic neuron
#     Panel(cd_data.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",legend=False,
#           yticks=True,xticks=True, xlim=(0, duration)),
#     # Panel(cd_data.segments[0].filter(name='gsyn_exc')[0],
#     #       ylabel="gsyn excitatory (mV)",legend=False,
#     #        yticks=True,xticks=True, xlim=(0, duration)),
#     # Panel(cd_data.segments[0].filter(name='gsyn_inh')[0],
#     #       ylabel="gsyn inhibitory (mV)",legend=False,
#     #        yticks=True,xticks=True, xlim=(0, duration)),
#     Panel(cd_data.segments[0].spiketrains,marker='.',
#           yticks=True,markersize=3,
#                  markerfacecolor='black', markeredgecolor='none',
#                  markeredgewidth=0,xticks=True, xlim=(0, duration)),
#     # Panel(input_data.segments[0].spiketrains, marker='.',
#     #       yticks=True, markersize=3,
#     #       markerfacecolor='black', markeredgecolor='none',
#     #       markeredgewidth=0, xticks=True, xlim=(0, duration))
# )

# psth_plot_8(plt, numpy.arange(len(t_spikes_combined)),t_spikes_combined , bin_width=0.25 / 1000.,
#             duration=duration/1000., title='psth input')
# psth_plot_8(plt, numpy.arange(len(cd_data.segments[0].spiketrains)),cd_data.segments[0].spiketrains , bin_width=0.25 / 1000.,
#             duration=duration/1000., title='psth output')
# title = "Izhikevich neuron"
for i in range(1,2):
    title = "LIF neuron_{}".format(i)
    plt.figure(title)
    spike_raster_plot_8(cd_data[i].segments[0].spiketrains,plt,duration/1000.,n_o+1,0.001,title=title,subplots=(3,1,1))
    mem_v = cd_data[i].segments[0].filter(name='v')
    cell_voltage_plot_8(mem_v, plt, duration, [],id=n_o/2,scale_factor=0.0001,title="",subplots=(3,1,2))
    plt.ylabel("membrane voltage (mV)")
    gsyn = cd_data[i].segments[0].filter(name='gsyn_exc')
    cell_voltage_plot_8(gsyn, plt, duration, [],scale_factor=0.0001,title="",subplots=(3,1,3))


#
# ch_spikes = cd_data.segments[0].spiketrains
#
# spike_raster_plot_8(ch_spikes,plt,duration/1000.,target_pop_size+1,0.001,title="cd pop activity")
# spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")
# cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.0001,title='cd pop')

# psth_plot_8(plt,numpy.arange(target_pop_size),cd_data.segments[0].spiketrains,bin_width=0.001,duration=duration/1000.,title="PSTH_CH")
# isi_t = [isi(spikes) for spikes in cd_data.segments[0].spiketrains]
# plt.figure("ISI")
# for i,neuron in enumerate(isi_t):
#     all_isi = [interval.item() for interval in neuron]
#     plt.subplot(target_pop_size/2,2,i+1)
#     plt.hist(all_isi)
#     plt.xlim((0,20))

plt.show()