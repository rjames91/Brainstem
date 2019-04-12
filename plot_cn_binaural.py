import numpy as np
from scipy.io import wavfile
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv

plot_spikes = True
plot_moc = True
plot_isi = False
plot_psth = False
# n_total = int(2. * n_fibres)
# Open the results
results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
test_index = 11
# results_file = "/cn_chirp_0s_1000an_fibres_0.1ms_timestep_65dB_1s_moc_True_lat_True.npz"
# results_file = "/cn_timit_0s_1000an_fibres_0.1ms_timestep_65dB_2s_moc_True_lat_True_{}.npz".format(test_index)
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_False.npz"
results_file = "/cn_tone_1000Hz_stereo_0s_100an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True_{}.npz".format(test_index)
n_ears = 2

# duration = binaural_audio_data[0].size/Fs
duration = 0.45
results_data = np.load(results_directory+results_file)
Fs = results_data['Fs']
t_spikes_split = results_data['t_spikes']
t_spikes_combined =[]
for ear_index in range(len(t_spikes_split)):
    t_spikes_combined.append([val for tup in zip(*t_spikes_split[ear_index]) for val in tup])
d_spikes_split = results_data['d_spikes']
d_spikes_combined =[]
for ear_index in range(len(d_spikes_split)):
    d_spikes_combined.append([val for tup in zip(*d_spikes_split[ear_index]) for val in tup])
b_spikes = results_data['b_spikes']
o_spikes = results_data['o_spikes']
moc_spikes = results_data['moc_spikes']
an_spikes = results_data['an_spikes']
moc_att = results_data['moc_att']

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
n_tds = 10
t_ds = np.logspace(np.log10(2),np.log10(200),n_tds)
n_dds = 10
d_ds = np.logspace(np.log10(2),np.log10(15),n_dds)

for ear_index in range(n_ears):
    neuron_title_list = ['t_stellate','d_stellate', 'bushy', 'octopus','moc','an']
    # neuron_list = [t_spikes_combined,d_spikes_combined, b_spikes, o_spikes,moc_spikes,an_spikes]
    neuron_list = [moc_spikes]
    # neuron_title_list = ["T stellate d = " + str(td) for td in t_ds]
    # neuron_list = t_spikes_split[ear_index]
    # neuron_title_list = ["D stellate d = " + str(dd) for dd in d_ds]
    # neuron_list = d_spikes_split[ear_index]

    for i, neuron_times in enumerate(neuron_list):
        non_zero_neuron_times = neuron_times[ear_index]
        mid_point = int(len(non_zero_neuron_times)/2.)
        psth_spikes = non_zero_neuron_times[mid_point-10:mid_point+10]

        if plot_spikes:
            plt.figure("spikes ear{} test {}".format(ear_index,test_index))
            # non_zero_neuron_times = neuron_times
            spike_raster_plot_8(non_zero_neuron_times, plt, duration, len(non_zero_neuron_times) + 1, 0.001,
                                title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1)
                                )  # ,filepath=results_directory)
        if plot_psth:
            plt.figure("psth ear{}".format(ear_index))
            # psth_spikes = [non_zero_neuron_times[mid_point]]
            psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25 / 1000.,
                        duration=duration,title=neuron_title_list[i],subplots=(len(neuron_list), 1, i + 1))
        if plot_isi:
            plt.figure("isi ear{}".format(ear_index))
            t_isi = [isi(spike_train) for spike_train in psth_spikes]
            hist_isi = []
            for neuron in t_isi:
                for interval in neuron:
                    if interval.item()<20:
                        hist_isi.append(interval.item())
            plt.subplot(len(neuron_list), 1, i + 1)
            plt.hist(hist_isi)
            # plt.xlim((0,20))
            # plt.ylim((0,100))

    if plot_moc:
        plt.figure("moc ear {} test {}".format(ear_index,test_index))

        for i,moc in enumerate(moc_att[ear_index]):
            if moc.min()==1:
                print "test {} ear {} channel {}".format(test_index,ear_index,i)
            t = np.linspace(0,duration*1000.,len(moc))
            plt.plot(t,moc)
        plt.ylabel("MOC attenuation")
        plt.xlabel("time (ms)")

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
neuron_list = [t_spikes]#,d_spikes,b_spikes,o_spikes]
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