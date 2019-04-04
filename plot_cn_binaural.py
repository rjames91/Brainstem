import numpy as np
from scipy.io import wavfile
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv

n_fibres = 300
duration = 2#3
n_total = int(6.66 * n_fibres)
# n_total = int(2. * n_fibres)
# Open the results
results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# results_file = "/cn_timit_3s_{}an_fibres_1.0ms_timestep_255_128percore.npz".format(n_fibres)
# results_file = "/cn_timit_{}s_{}an_fibres_1.0ms_timestep_filtered_edges_255_128percore_n_total60000.0.npz".format(duration,n_fibres)
# results_file = "/cn_timit_{}s_{}an_fibres_1.0ms_timestep_filtered_edges_255_255percore_tx_offset_n_total{}.npz".format(duration,n_fibres,n_total)
# results_file = "/cn_yes_2s_{}an_fibres_1.0ms_timestep_filtered_edges.npz".format(n_fibres)
# results_file = "/cn_timit_2s_300an_fibres_1.0ms_timestep_filtered_edges.npz"
# results_file = "/cn_tone_1_26s_3000an_fibres_0.1ms_timestep_filtered_edges_255_255percore_tx_offset_n_total19980_ts200.npz"
results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_65dB_0s_d_200_moc_True_lat_True.npz"
# ear_results_file = "/spinnakear_timit_{}s_30dB_{}fibres.npz".format(duration,n_fibres)
# ear_results_file = "/spinnakear_timit_{}s_30dB_{}fibres.npz".format(duration,n_fibres)
# # ear_results_file = "/spinnakear_yes_2s_30dB_{}fibres.npz".format(n_fibres)
# ear_results_data = np.load(results_directory+ear_results_file)
# binaural_audio_data = ear_results_data['audio_data']
# binaural_scaled_times = ear_results_data['scaled_times']
# if binaural_scaled_times.shape[0]>1:
#     n_ears =2
# else:
#     n_ears = 1
n_ears = 2

# duration = binaural_audio_data[0].size/Fs
duration = 0.5
results_data = np.load(results_directory+results_file)
Fs = results_data['Fs']
t_spikes = results_data['t_spikes']
d_spikes = results_data['d_spikes']
b_spikes = results_data['b_spikes']
o_spikes = results_data['o_spikes']
an_spikes = results_data['an_spikes']

# left_t_spike_trains = [spike_train for spike_train in t_spikes[0] if len(spike_train)>0]
# half_point = len(left_t_spike_trains)/2
# t_isi = [isi(spike_train) for spike_train in left_t_spike_trains[:]]#[half_point-10:half_point+10]]
# cvs = [cv(isi) for isi in t_isi]
#
# fig,axs=plt.subplots(2,1)
# axs[1].hist(cvs)
# axs[1].set_title("Coefficient of variation")
# # plt.hist(cvs)
# all_isi=[]
# for i,neuron in enumerate(t_isi):
#     for interval in neuron:
#         if interval<20:
#             all_isi.append(interval.item())
#     # all_isi = [interval.item() for interval in neuron]
#     # plt.subplot(target_pop_size/2,2,i+1)
# axs[0].hist(all_isi)
# # axs[0].xlim((0,20))
# axs[0].set_title("ISI left")

for ear_index in range(n_ears):
    neuron_title_list = ['t_stellate', 'd_stellate', 'bushy', 'octopus','an']
    neuron_list = [t_spikes, d_spikes, b_spikes, o_spikes,an_spikes]

    for i, neuron_times in enumerate(neuron_list):
        # plt.figure("spikes ear{}".format(ear_index))
        non_zero_neuron_times = neuron_times[ear_index]#[spikes for spikes in neuron_times[ear_index] if len(spikes)>0]#
        # spike_raster_plot_8(non_zero_neuron_times, plt, duration, len(non_zero_neuron_times) + 1, 0.001,
        #                     title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1)
        #                     )  # ,filepath=results_directory)
        # plt.figure("psth ear{}".format(ear_index))
        mid_point = int(len(non_zero_neuron_times)/2.)
        psth_spikes = non_zero_neuron_times[mid_point-10:mid_point+10]
        # psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25 / 1000.,
        #             duration=duration,title=neuron_title_list[i],subplots=(len(neuron_list), 1, i + 1))
        plt.figure("isi ear{}".format(ear_index))
        t_isi = [isi(spike_train) for spike_train in psth_spikes]
        hist_isi = []
        for neuron in t_isi:
            for interval in neuron:
                if interval.item()<20:
                    hist_isi.append(interval.item())
        plt.subplot(len(neuron_list), 1, i + 1)
        plt.hist(hist_isi)
        plt.xlim((0,20))
        plt.ylim((0,100))

plt.show()

# o_results_file = '/octopus_timit_3s_{}an_fibres_1.0ms_timestep_filtered_edges_255_255percore_tx_offset_n_total{}.npz'.format(n_fibres,n_total)
# o_results_data = np.load(results_directory+o_results_file)
# o_spikes = o_results_data['o_spikes']

n_t = int(n_total * 2. / 3 * 24. / 89)
n_d = int(n_total * 1. / 3 * 24. / 89)
n_b = int(n_total * 55. / 89)  # number_of_inputs#
n_o = int(n_total * 10. / 89.)
population_size_dict = {'t_stellate':n_t,
                        'd_stellate':n_d,
                        'bushy':n_b,
                        'octopus':n_o
}
pop_size = max([n_fibres,n_d,n_t,n_b,n_o])
# pop_size = max([n_fibres,n_o])
neuron_title_list = ['t_stellate','d_stellate','bushy','octopus']
neuron_list = [t_spikes,d_spikes,b_spikes,o_spikes]
# neuron_title_list = ['octopus']
# neuron_list = [o_spikes]
ear_list = ['left','right']
colour_list = ['blue','red']
for ear_index in range(n_ears):
    # plt.figure('cn spikes '+ear_list[ear_index]+' ear')
    plt.figure('cn spikes',figsize=(16,9))
    plt.tight_layout()
    for i,neuron_times in enumerate(neuron_list):
        # non_zero_neuron_times = [spikes for spikes in neuron_times[ear_index] if len(spikes)>0]
        ids = np.linspace(0, pop_size-1,population_size_dict[neuron_title_list[i]], dtype=int)
        non_zero_neuron_times = [spikes for j,spikes in enumerate(neuron_times[ear_index]) if j in ids]
        spike_raster_plot_8(non_zero_neuron_times, plt, duration, len(non_zero_neuron_times) + 1, 0.001,
                            title=neuron_title_list[i],markersize=1,marker_colour=colour_list[ear_index],
                            alpha=0.5,subplots=(len(neuron_list),1,i+1),legend_strings=ear_list)#,filepath=results_directory)

plt.show()