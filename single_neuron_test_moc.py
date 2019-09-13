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

octopus_params_cond = {'cm': 1.,#57.,  # nF
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -55.,#-20.,#-10.,#-35.,#-55.1,#
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


izk_class_1_params = {
               'a':0.03,
               'b':-2,
               'c':-50,
               'd':80,
               'u':0,
               'tau_syn_E': 0.94,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -70.0,
}

d_stellate_izk_class_2_params = {
               'a':0.2,
               'b':0.26,
               'c':-65,
               'd':0,
               'u':-15,
               'tau_syn_E':4.88,
               'e_rev_E': -30.,
               'tau_syn_I':4.,
               'v': -63.0,
}

d_stellate_izk_class_1_params = {
               'a':0.02,
               'b':-0.1,
               'c':-55,
               'd':6,
               'u':10,
               'tau_syn_E':4.88,
               'e_rev_E': -30.,
               'tau_syn_I':4.,
               'v': -63.0,
}

IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                   'v': -75,
                   'u': 10.,#0.,
                   }

octopus_params_cond_izh = {
               # 'a':0.02,
               # 'b':-0.1,
               # 'c':-55,
               # 'd':6,
               # 'u':10,
                'a':0.02,
                'b':0.25,
                'c':-65,
                'd':4,
                'u':-15,
               'tau_syn_E': 0.2,#0.35,#2.5,#
               'e_rev_E': -55.,#-25.,#-10.,#-35.,#-55.1,#
               'v': -70.,
               }

t_stellate_izk_class_2_params = {
               'a':0.5,#0.02,#0.2,
               'b':0.26,
               'c':-65,
               'd':10,#400,#220,#12,#vary between 12 and 220 for CRs 100-500Hz
               'u':0,#-15,
               'tau_syn_E': 0.94,#3.0,#2.5,#
               # 'tau_syn_E':4. ,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -63.0,
               # 'i_offset':-5.
}

bushy_params_cond = {#'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               #'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 2.,#2.5,#
               #'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.,#-70.0,
               'v_rest': -60.,
               'v_thresh': -40.
               }
moc_tonic_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd':6,
    'u': -10,
    'v': -70.0,
}
moc_class_2_params = {
    'a': 0.02,#0.2,
    'b': 0.26,
    'c': -65,
    'd': 0,
    'u': -15,
    'v': -65,
    'tau_syn_E':3.,
}

moc_rs_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8,
    'u': -15,
    'v': -70,
    # 'tau_syn_E': 0.5,
}

moc_lif_params = {
    'tau_syn_E': 2.,
    # 'tau_m':1.
}

moc_lts_params = {
    'a': 0.02,
    'b': 0.25,
    'c': -65,
    'd':2,
    'u': -10,
    'v': -65,
}
moc_lts_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 1.,
    'u': -10,
    'v': -65,
}
moc_lts_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 0.5,
    'u': -10,
    'v': -65,
    'tau_syn_E': 2.
}
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_timit_0s_1000an_fibres_0.1ms_timestep_50dB_2s_moc_True_lat_True.npz"
results_file = "/cn_tone_1000Hz_stereo_1s_1000an_fibres_0.1ms_timestep_50dB_1s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_3000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_False.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True_{}.npz".format(test_index)

results_data = np.load(input_directory+results_file)
t_data_split = results_data['t_data']

t_spikes_split = [split.segments[0].spiketrains for split in t_data_split[0]]
t_combined = split_population_data_combine(t_data_split,['spikes'])
t_spikes_combined = t_combined['spikes'][0]
stimulus = results_data['stimulus']
Fs = results_data['Fs']

duration = 60.#1000*len(stimulus[0])/Fs


n_tds = len(t_spikes_split)
n_ears = 2
n_fibres = 1000
n_total = int(2.4 * n_fibres)

#ratios taken from campagnola & manis 2014 mouse
n_t = int(n_total * 2./3 * 24./89)
n_d = int(n_total * 1./3 * 24./89)
n_b = int(n_total * 55./89)#number_of_inputs#
n_o = int(n_total * 10./89.)
n_moc =360
n_sub_t=int(n_t/n_tds)
pop_size = max([n_fibres,n_d,n_t,n_b,n_o,n_moc])

w2s_moc = 30.#0.2#0.75#0.05  # 0.75
# n_t_moc_connections = RandomDistribution('uniform', [5, 10])
av_t_moc_connections = 10  # int(np.ceil(float(n_t)/n_moc))
n_t_moc_connections = RandomDistribution('normal_clipped',
                                         [av_t_moc_connections, 0.1 * av_t_moc_connections, 0,
                                          av_t_moc_connections * 2.])
av_t_moc = w2s_moc / av_t_moc_connections  # 9.
t_moc_weight = RandomDistribution('normal_clipped', [av_t_moc, 0.1 * av_t_moc, 0, av_t_moc * 2.])
t_mocc_master, max_dist = normal_dist_connection_builder(n_t, n_moc, RandomDistribution,
                                                         conn_num=n_t_moc_connections, dist=1.,
                                                         sigma=float(pop_size) / n_moc,
                                                         conn_weight=t_moc_weight, get_max_dist=True,
                                                         normalised_space=pop_size)
t_mocc_list = [[] for _ in range(n_tds)]
for (pre, post, w, d) in t_mocc_master:
    i = np.remainder(pre, n_tds)
    t_mocc_list[i].append((int((float(pre) / n_t) * n_sub_t + 0.5), post, w, d))

target_pop_size =360

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=0.1)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#================================================================================================
# Populations
#================================================================================================
input_pops=[]
for i,in_spikes in enumerate(t_spikes_split):
    n_inputs = len(in_spikes)
    input_pops.append(sim.Population(n_inputs, sim.SpikeSourceArray(spike_times=in_spikes), label="an_pop_input {}".format(i)))

# input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_pop_input")
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params,label="fixed_weight_scale")
# cd_pop = sim.Population(target_pop_size,sim.IF_curr_exp,one_to_one_cond_params,label="fixed_weight_scale")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,moc_class_2_params,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(target_pop_size,sim.IF_cond_exp,moc_lif_params,label="fixed_weight_scale_cond")
cd_pop = sim.Population(target_pop_size,sim.Izhikevich,moc_lts_params,label="moc")
# cd_pop = sim.Population(1,sim.IF_curr_exp,on_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =
# input_pop.record("spikes")
cd_pop.record("all")
#================================================================================================
# Projections
#================================================================================================
for i, t_mocc_l in enumerate(t_mocc_list):
    if len(t_mocc_l) > 0:
        sim.Projection(input_pops[i], cd_pop,sim.FromListConnector(t_mocc_l), synapse_type=sim.StaticSynapse())

sim.run(duration)

cd_data = cd_pop.get_data()
# input_data = input_pop.get_data()

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
    # Panel(cd_data.segments[0].spiketrains,marker='.',
    #       yticks=True,markersize=1,
    #              markerfacecolor='black', markeredgecolor='none',
    #              markeredgewidth=0,xticks=True, xlim=(0, duration)),
    # Panel(input_data.segments[0].spiketrains, marker='.',
    #       yticks=True, markersize=3,
    #       markerfacecolor='black', markeredgecolor='none',
    #       markeredgewidth=0, xticks=True, xlim=(0, duration))
)

psth_plot_8(plt, numpy.arange(len(t_spikes_combined)),t_spikes_combined , bin_width=0.25 / 1000.,
            duration=duration/1000., title='psth input')
psth_plot_8(plt, numpy.arange(len(cd_data.segments[0].spiketrains)),cd_data.segments[0].spiketrains , bin_width=0.25 / 1000.,
            duration=duration/1000., title='psth output')
plt.figure("spikes")
spike_raster_plot_8(t_spikes_combined,plt,duration/1000.,n_t+1,0.001,title="input activity",subplots=(2,1,1),markersize=1)
spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,n_moc+1,0.001,title="moc activity",subplots=(2,1,2),markersize=1)

# mem_v = cd_data.segments[0].filter(name='v')
# # cell_voltage_plot_8(mem_v, plt, duration, [],id=599,scale_factor=0.001,title='cd pop')
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