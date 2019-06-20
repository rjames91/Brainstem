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
n_dds = 10
# d_ds = np.logspace(np.log10(4),np.log10(16),n_dds)
d_ds = np.logspace(np.log10(4),np.log10(50),n_dds)
d_stellate_izk_class_1_params = {
               'a':0.02,
               'b':-0.1,
               'c':-55,
               'd':4,
               'u':10,
               'tau_syn_E':4.88,
               'e_rev_E': 0.,
               'tau_syn_I':4.,
               'v': -63.0,
}

dB = 60#20
n_fibres = 1000
freq=1000

input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

# results_file = "/t_stellate_tone_{}Hz_0s_{}an_fibres_0.1ms_timestep_{}dB_0s_moc_False_lat_False.npz".format(freq,n_fibres,dB)
# results_file = "/t_stellate_tone_1000Hz_stereo_0s_3000an_fibres_0.1ms_timestep_65dB_0s_moc_False_lat_False.npz"
results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_60dB_0s_moc_True_lat_True.npz"

results_data = np.load(input_directory+results_file)
an_spikes = results_data['an_spikes'][0]

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
n_sub_d=int(n_d/n_dds)
pop_size = max([n_fibres,n_d,n_t,n_b,n_o,n_moc])

d_pops = [[] for __ in range(n_dds)]
d_spikes = [[] for __ in range(n_dds)]
d_data = [[] for __ in range(n_dds)]

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
for dd in range(n_dds):
    d_stellate_izk_class_1_params['d'] = d_ds[dd]
    d_pops[dd] = sim.Population(n_sub_d, sim.extra_models.Izhikevich_cond, d_stellate_izk_class_1_params,
                                           label="d_stellate_fixed_weight_scale_cond")
    # t_stellate_lif_params['tau_m'] = t_ds[td]
    # t_stellate_lif_params['v_thresh'] = t_ths[td]
    # t_pops[td] = sim.Population(n_sub_t, sim.IF_cond_exp, t_stellate_lif_params,
    #                                        label="t_stellate_fixed_weight_scale_cond")

    d_pops[dd].record("all")
# cd_pop = sim.Population(target_pop_size,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="fixed_weight_scale_cond")

# cd_pop.record("all")
#================================================================================================
# Projections
#================================================================================================
# connection_dicts_file = np.load(input_directory+'/cn_{}an_fibres_{}ears_connectivity.npz'.format
#                         (n_fibres,n_ears))
# connection_dicts = connection_dicts_file['connection_dicts']

n_an_d_connections = RandomDistribution('normal_clipped',
                                        [60., 5., 11., 88.])  # estimation of upper and lower bounds Manis 2017
w2s_d = 0.75  # 0.3#1.#0.5#
av_an_d = w2s_d / 88.  # w2s_d/88.
# an_d_weight = RandomDistribution('uniform',[0,av_an_d])
an_d_weight = RandomDistribution('normal_clipped', [av_an_d, 0.1 * av_an_d, 0, av_an_d * 2.])
an_d_master, max_dist = normal_dist_connection_builder(n_fibres, n_d, RandomDistribution,
                                                       conn_num=n_an_d_connections, dist=1.,
                                                       sigma=pop_size / 15., conn_weight=an_d_weight,
                                                       delay=0.1, normalised_space=pop_size, get_max_dist=True)
an_d_list = [[] for _ in range(n_dds)]
for (pre, post, w, d) in an_d_master:
    i = np.remainder(post, n_dds)
    an_d_list[i].append((pre, int((float(post) / n_d) * n_sub_d + 0.5), w, d))

for i, source_l in enumerate(an_d_list):
    if len(source_l) > 0:
        sim.Projection(input_pop, d_pops[i], sim.FromListConnector(source_l),
                           synapse_type=sim.StaticSynapse())

# an_t_list = connection_dicts[0]['an_t_list']

for i, source_l in enumerate(an_d_list):
    if len(source_l) > 0:
            sim.Projection(input_pop, d_pops[i], sim.FromListConnector(source_l),
                           synapse_type=sim.StaticSynapse())

duration = 100#max(input_spikes[0])

sim.run(duration)

for i, pop in enumerate(d_pops):
    d_data[i] = pop.get_data()
    # t_spikes[i] = t_data[i].segments[0].spiketrains

sim.end()

for i,cd_data in enumerate(d_data):
    plt.figure('spikes')
    non_zero_neuron_times = cd_data.segments[0].spiketrains
    spike_raster_plot_8(non_zero_neuron_times, plt, duration/1000., len(non_zero_neuron_times) + 1, 0.001,
                         markersize=1, subplots=(len(d_data), 1, i + 1),title=str(d_ds[i])
                        )
    plt.figure("psth")
    psth_spikes = non_zero_neuron_times[:]
    psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=1e-3,
                duration=duration/1000.,title=str(d_ds[i]),subplots=(len(d_data), 1, i + 1),ylim=500)

    plt.figure("v_mem")
    mem_v = cd_data.segments[0].filter(name='v')
    cell_voltage_plot_8(mem_v, plt, duration, [],id=int(len(non_zero_neuron_times)/2),scale_factor=0.0001,title="",subplots=(len(d_data), 1, i + 1))
    # plt.ylabel("membrane voltage (mV)")

    plt.figure("g_syn")
    g_syn = cd_data.segments[0].filter(name='gsyn_exc')
    cell_voltage_plot_8(g_syn, plt, duration, [],id=int(len(non_zero_neuron_times)/2),scale_factor=0.0001,title="",subplots=(len(d_data), 1, i + 1))


    plt.figure("isi")
    t_isi = [isi(spike_train) for spike_train in psth_spikes]
    hist_isi = []
    for neuron in t_isi:
        for interval in neuron:
            if interval.item()<20:
                hist_isi.append(interval.item())
    plt.subplot(len(d_data), 1, i + 1)
    plt.hist(hist_isi,bins=100)
    plt.xlim((0,20))
    plt.figure("CV")
    cvs = [cv(interval) for interval in t_isi if len(interval) > 0]
    plt.subplot(len(d_data), 1, i + 1)
    plt.hist(cvs)  # ,bins=100)
    plt.xlim((0, 2))

spike_raster_plot_8(an_spikes, plt, duration/1000., len(an_spikes) + 1, 0.001,
                     markersize=1, title= "an spikes" )

mid_point = int(len(an_spikes)/2)
psth_spikes = an_spikes[mid_point-100:mid_point+100]
psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25e-3,
            duration=duration/1000.,ylim=None,title='psth an')

an_count = 0
for neuron in an_spikes:
    an_count+=len(neuron)
print "an spike count = {}".format(an_count)
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