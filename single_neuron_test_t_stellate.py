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
t_stellate_izk_class_2_params = {
               'a':0.4,#0.2,#0.02,#
               'b':0.26,
               'c':-65,
               # 'd':8,
               'u':0,#-15,
               'tau_syn_E':4.0,#2.,#0.94,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -63.0,
}
t_stellate_izk_class_1_params={
    'a':0.02,
    'b':-0.1,
    'c':-55,
    'd':6,


}

t_stellate_lif_params ={'tau_syn_E': 0.94}
dB = 50#20
n_fibres = 1000
freq=1000

input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

# results_file = "/t_stellate_tone_{}Hz_0s_{}an_fibres_0.1ms_timestep_{}dB_0s_moc_False_lat_False.npz".format(freq,n_fibres,dB)
# results_file = "/t_stellate_tone_1000Hz_stereo_0s_3000an_fibres_0.1ms_timestep_65dB_0s_moc_False_lat_False.npz"
# results_file = "/cn_tone_1000Hz_stereo_1s_1000an_fibres_0.1ms_timestep_80dB_1s_moc_True_lat_True.npz"
results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True.npz"

results_data = np.load(input_directory+results_file)
sg_spikes = [data.segments[0].spiketrains for data in results_data['sg_data']]
an_spikes=sg_spikes[0]
n_fibres = len(an_spikes)

n_tds = 10
# t_ds = np.linspace(1.,20.,n_tds)
# t_ths = np.linspace(-60.,-40.,n_tds)
t_ds = np.logspace(np.log10(1),np.log10(150),n_tds)
# t_ds = np.linspace(1e-6,150,n_tds)
# plt.figure()
# plt.plot(t_ds)
# plt.show()
n_ears = 2
n_total = int(2.4 * n_fibres)
#ratios taken from campagnola & manis 2014 mouse
n_t = int(n_total * 2./3 * 24./89)
n_d = int(n_total * 1./3 * 24./89)
n_b = int(n_total * 55./89)#number_of_inputs#
n_o = int(n_total * 10./89.)
n_moc =360
n_sub_t=int(n_t/n_tds)
pop_size = max([n_fibres,n_d,n_t,n_b,n_o,n_moc])

t_pops = [[] for __ in range(n_tds)]
t_spikes = [[] for __ in range(n_tds)]
t_data = [[] for __ in range(n_tds)]

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=0.1)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,256)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(n_fibres,sim.SpikeSourceArray(spike_times=an_spikes),label="an_pop_input")
for td in range(n_tds):
    t_stellate_izk_class_2_params['d'] = t_ds[td]
    t_pops[td] = sim.Population(n_sub_t, sim.Izhikevich, t_stellate_izk_class_2_params,
                                           label="t_stellate")#_fixed_weight_scale")
    # t_stellate_lif_params['tau_m'] = t_ds[td]
    # t_stellate_lif_params['v_thresh'] = t_ths[td]
    # t_pops[td] = sim.Population(n_sub_t, sim.IF_cond_exp, t_stellate_lif_params,
    #                                        label="t_stellate_fixed_weight_scale_cond")

    t_pops[td].record("all")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="fixed_weight_scale_cond")

# cd_pop.record("all")
#================================================================================================
# Projections
#================================================================================================
# connection_dicts_file = np.load(input_directory+'/cn_{}an_fibres_{}ears_connectivity.npz'.format
#                         (n_fibres,n_ears))
# connection_dicts = connection_dicts_file['connection_dicts']

w2s_t = 12.#50.#15.#0.4#1.6#0.4#0.1# 0.8#0.5#0.25#0.1#0.3#0.1#0.7
n_an_t_connections = RandomDistribution('uniform', [4., 6.])
av_an_t = w2s_t / 5.
# an_t_weight = RandomDistribution('uniform',[0,av_an_t*2])
# an_t_weight = RandomDistribution('uniform',[av_an_t/5.,av_an_t*2])
an_t_weight = RandomDistribution('normal_clipped', [av_an_t, 0.1 * av_an_t, 0, av_an_t * 2.])
an_t_master, max_dist = normal_dist_connection_builder(n_fibres, n_t, RandomDistribution,
                                                       conn_num=n_an_t_connections,
                                                       dist=1., sigma=1., conn_weight=an_t_weight,
                                                       normalised_space=pop_size, get_max_dist=True,delay=0.1)
an_t_list = [[] for _ in range(n_tds)]
for (pre, post, w, d) in an_t_master:
    i = np.remainder(post, n_tds)
    an_t_list[i].append((pre, int((float(post) / n_t) * n_sub_t + 0.5), w, d))

# an_t_list = connection_dicts[0]['an_t_list']

for i, source_l in enumerate(an_t_list):
    if len(source_l) > 0:
            sim.Projection(input_pop, t_pops[i], sim.FromListConnector(source_l),
                           synapse_type=sim.StaticSynapse())

duration = 100#max(input_spikes[0])

sim.run(duration)

for i, pop in enumerate(t_pops):
    t_data[i] = pop.get_data()
    # t_spikes[i] = t_data[i].segments[0].spiketrains

sim.end()

for i,cd_data in enumerate(t_data):
    plt.figure('spikes')
    non_zero_neuron_times = cd_data.segments[0].spiketrains
    # spike_raster_plot_8(non_zero_neuron_times, plt, duration/1000., len(non_zero_neuron_times) + 1, 0.001,
    spike_raster_plot_8([non_zero_neuron_times[int(len(non_zero_neuron_times)/2)]], plt, duration/1000.,  2, 0.001,
                         markersize=1, subplots=(len(t_data), 1, i + 1),title=str(t_ds[i])
                        )
    # plt.figure("psth")
    psth_spikes = non_zero_neuron_times[:]
    # psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=1e-3,
    #             duration=duration/1000.,title=str(t_ds[i]),subplots=(len(t_data), 1, i + 1),ylim=500)

    plt.figure("v_mem")
    mem_v = cd_data.segments[0].filter(name='v')
    cell_voltage_plot_8(mem_v, plt, duration, [],id=int(len(non_zero_neuron_times)/2),scale_factor=0.0001,title="",subplots=(len(t_data), 1, i + 1))
    # plt.ylabel("membrane voltage (mV)")

    plt.figure("g_syn")
    g_syn = cd_data.segments[0].filter(name='gsyn_exc')
    cell_voltage_plot_8(g_syn, plt, duration, [],id=int(len(non_zero_neuron_times)/2),scale_factor=0.0001,title="",subplots=(len(t_data), 1, i + 1))


    plt.figure("isi")
    t_isi = [isi(spike_train) for spike_train in psth_spikes]
    hist_isi = []
    for neuron in t_isi:
        for interval in neuron:
            if interval.item()<20:
                hist_isi.append(interval.item())
    plt.subplot(len(t_data), 1, i + 1)
    plt.hist(hist_isi,bins=100)
    plt.xlim((0,20))
    # plt.figure("CV")
    # cvs = [cv(interval) for interval in t_isi if len(interval) > 0]
    # plt.subplot(len(t_data), 1, i + 1)
    # plt.hist(cvs)  # ,bins=100)
    # plt.xlim((0, 2))

# spike_raster_plot_8(an_spikes, plt, duration/1000., len(an_spikes) + 1, 0.001,
#                      markersize=1, title= "an spikes" )
#
# mid_point = int(len(an_spikes)/2)
# psth_spikes = an_spikes[mid_point-100:mid_point+100]
# psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25e-3,
#             duration=duration/1000.,ylim=None,title='psth an')
#
# an_count = 0
# for neuron in an_spikes:
#     an_count+=len(neuron)
# print "an spike count = {}".format(an_count)
# mem_v = cd_data.segments[0].filter(name='v')
# # cell_voltage_plot_8(mem_v, plt, duration, [],id=599,scale_factor=0.001,title='cd pop')

# psth_plot_8(plt, numpy.arange(len(t_spikes_combined)),t_spikes_combined , bin_width=0.25 / 1000.,
#             duration=duration/1000., title='psth input')
# psth_plot_8(plt, numpy.arange(len(cd_data.segments[0].spiketrains)),cd_data.segments[0].spiketrains , bin_width=0.25 / 1000.,
#             duration=duration/1000., title='psth output')
# spike_raster_plot_8(t_spikes_combined,plt,duration/1000.,n_t+1,0.001,title="input activity")

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