import numpy as np
from scipy.io import wavfile,loadmat
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv
import quantities as pq

plot_spikes = True
plot_moc = True
plot_isi = False
plot_psth = False
plot_abr = False
plot_stim =True
# n_total = int(2. * n_fibres)
# Open the results
results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
test_index = 1
# results_file = "/cn_chirp_0s_1000an_fibres_0.1ms_timestep_65dB_1s_moc_True_lat_True_{}.npz".format(test_index)
# results_file = "/cn_timit_0s_1000an_fibres_0.1ms_timestep_65dB_2s_moc_True_lat_True_{}.npz".format(test_index)
# results_file = "/cn_timit_0s_1000an_fibres_0.1ms_timestep_65dB_2s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_1s_moc_True_lat_True.npz"
# results_file = "/cn_timit_0s_3000an_fibres_0.1ms_timestep_65dB_2s_moc_True_lat_True.npz"
# results_file = "/cn_tone_stereo_1000Hz_0s_3000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_3000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_1s_1000an_fibres_0.1ms_timestep_100dB_1s_moc_True_lat_True.npz"
# results_file = "/cn_click_0s_1000an_fibres_0.1ms_timestep_80dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_click_0s_1000an_fibres_0.1ms_timestep_60dB_0s_moc_True_lat_True.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_50dB_0s_moc_True_lat_True.npz"
results_file = "/cn_timit_0s_1000an_fibres_0.1ms_timestep_65dB_3s_moc_True_lat_True.npz"

# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_100dB_0s_moc_True_lat_False.npz"
# results_file = "/cn_tone_1000Hz_stereo_0s_1000an_fibres_0.1ms_timestep_65dB_0s_moc_True_lat_True_{}.npz".format(test_index)
n_ears = 2

# duration = binaural_audio_data[0].size/Fs
recording_vars = ['spikes']

results_data = np.load(results_directory+results_file)
Fs = results_data['Fs']

sg_spikes = [data.segments[0].spiketrains for data in results_data['sg_data']]
t_data_split = results_data['t_data']
t_combined = split_population_data_combine(t_data_split,recording_vars)
t_spikes_combined = t_combined['spikes']
d_data_split = results_data['d_data']
d_combined = split_population_data_combine(d_data_split,recording_vars)
d_spikes_combined = d_combined['spikes']
b_spikes = [data.segments[0].spiketrains for data in results_data['b_data']]
o_spikes = [data.segments[0].spiketrains for data in results_data['o_data']]
moc_spikes = [data.segments[0].spiketrains for data in results_data['moc_data']]
an_spikes = [data['spikes'] for data in results_data['ear_data']]
moc_att = [data['moc'] for data in results_data['ear_data']]
onset_times = results_data['onset_times'][0]
stimulus = results_data['stimulus']

if 'v' in recording_vars:
    sg_mem_v = [data.segments[0].filter(name='v') for data in results_data['sg_data']]
    b_mem_v = [data.segments[0].filter(name='v') for data in results_data['b_data']]
    o_mem_v = [data.segments[0].filter(name='v') for data in results_data['o_data']]
    moc_mem_v = [data.segments[0].filter(name='v') for data in results_data['moc_data']]
    t_mem_v = t_combined['v']
    d_mem_v = d_combined['v']

duration = len(stimulus[0])/Fs

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
t_ds = np.logspace(np.log10(10),np.log10(500),n_tds)
n_dds = 10
d_ds = np.logspace(np.log10(4),np.log10(16),n_dds)

# ic_matlab = loadmat('./ic_spikes.mat')['ICoutput']
# ic_spikes_split = [ic_matlab[:10],ic_matlab[10:20],ic_matlab[20:]]
# ic_spikes_orig=[val for tup in zip(*ic_spikes_split) for val in tup]
# ic_spikes = []
# for neuron in ic_spikes_orig:
    # ic_spikes.append(np.nonzero(neuron)[0]*(1000/48e3))
for ear_index in range(n_ears):
    neuron_title_list = ['t_stellate','d_stellate', 'bushy', 'octopus','moc','an']
    neuron_list = [t_spikes_combined,d_spikes_combined, b_spikes, o_spikes,moc_spikes,sg_spikes]
    # neuron_title_list = ['an','bushy', 'octopus','moc']
    # neuron_list = [sg_spikes,b_spikes,o_spikes]#,moc_spikes]
    # neuron_title_list = ['t_stellate','d_stellate','moc','an']
    # neuron_list = [t_spikes_combined,d_spikes_combined,moc_spikes]#,an_spikes]
    # neuron_title_list = ['octopus','d','an']
    # neuron_list = [o_spikes,d_spikes_combined,an_spikes]
    # neuron_title_list = ["T stellate d = " + str(td) for td in t_ds]
    # neuron_list = t_spikes_split[ear_index]
    # neuron_title_list = ["D stellate d = " + str(dd) for dd in d_ds]
    # neuron_list = d_spikes_split[ear_index]
    abrs =[]

    for i, neuron_times in enumerate(neuron_list):
        non_zero_neuron_times = np.flipud(neuron_times[ear_index])
        # non_zero_neuron_times = np.flipud(neuron_times)
        mid_point = int(len(non_zero_neuron_times)*0.5)
        # psth_spikes = non_zero_neuron_times[:]
        psth_spikes = non_zero_neuron_times[mid_point-10:mid_point+10]
        # psth_spikes = [non_zero_neuron_times[mid_point]]
        # psth_spikes = repeat_test_spikes_gen(non_zero_neuron_times,mid_point,[onset_times[ear_index]],test_duration_ms=75)[0]

        if plot_spikes:
            plt.figure("spikes ear{} test {}".format(ear_index,test_index))
            spike_raster_plot_8(non_zero_neuron_times, plt, duration, len(non_zero_neuron_times) + 1, 0.001,
            # spike_raster_plot_8(psth_spikes, plt, duration, len(psth_spikes) + 1, 0.001,
                                title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1))  # ,filepath=results_directory)

        if plot_psth:
            plt.figure("psth ear{}".format(ear_index))
            psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25e-3,
                        duration=duration,title=neuron_title_list[i],subplots=(len(neuron_list), 1, i + 1),ylim=None)

        if plot_isi:
            plt.figure("isi ear{}".format(ear_index))
            t_isi = [isi(spike_train) for spike_train in psth_spikes]
            hist_isi = []
            for neuron in t_isi:
                for interval in neuron:
                    if interval.item()<20:
                        hist_isi.append(interval.item())
            plt.subplot(len(neuron_list), 1, i + 1)
            plt.hist(hist_isi,bins=100)
            plt.xlim((0,20))

            plt.figure("CV {}".format(ear_index))
            cvs = [cv(interval) for interval in t_isi if len(interval)>0]
            plt.subplot(len(neuron_list), 1, i + 1)
            plt.hist(cvs)#,bins=100)
            plt.xlim((0, 2))
            # plt.ylim((0,100))

    if plot_moc:
        plt.figure("moc ear {} test {}".format(ear_index,test_index))
        n_channels = len(moc_att[ear_index])
        # a = np.sum(moc_att[ear_index][::n_channels/10],axis=0)
        # a = moc_att[ear_index][n_channels/2]
        # plt.plot(a)
        # print "ear {} zero moc spikes {}".format(ear_index,len(np.where(a<1)[0]))
        for i,moc in enumerate(moc_att[ear_index][::n_channels/10]):
        # for i,moc in enumerate(moc_att[ear_index][::1]):
        # for i,moc in enumerate([moc_att[ear_index][n_channels/20]]):
            if moc.min()==1:
                print "test {} ear {} channel {}".format(test_index,ear_index,i)
            if 1:#moc.min()<0.9:
                t = np.linspace(0,duration*1000.,len(moc))
                plt.plot(t,moc)
                # plt.plot(moc)
        plt.ylabel("MOC attenuation")
        plt.xlabel("time (ms)")

    if 0:#plot_psth:
        mid_point = int(len(an_spikes[ear_index]) / 2)
        psth_spikes = sg_spikes[ear_index][:]#[mid_point - 100:mid_point + 100]
        psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25e-3,
                    duration=duration, ylim=1000, title='psth an')

    if plot_abr:
        # # mem_vs=[sg_mem_v,b_mem_v,t_mem_v,d_mem_v,o_mem_v,moc_mem_v]
        # # location_weights = [1.0, 0.04, 0.04, 0.04, 0.04, 0.01]
        # # v_rests = [-65., -65., -63., -63., -70., -65.]
        # mem_vs=[sg_mem_v,b_mem_v]#,o_mem_v]
        location_weights = [1.0, 0.03, 0.03,1.]
        # v_rests = [-65., -65., -70.]
        # # mem_vs=[sg_mem_v,b_mem_v]
        # combined_v = []
        # for i,mem_v in enumerate(mem_vs):
        #     combined_v.append((mem_v[ear_index][0]-v_rests[i]*pq.mV)*location_weights[i])
        #     # combined_v.append((mem_v[ear_index][0]-v_rests[i]*pq.mV))
        #     # abr,abr_time = abr_mem_v(mem_v[ear_index],duration*1000,ref_v=v_rests[i])
        #     # abrs.append(abr)
        # stacked_v = np.hstack(tuple(combined_v))
        # x = np.linspace(0, (duration*1000), len(stacked_v))
        # # reference_voltage = ref_v * pq.mV
        # abr = np.sum(stacked_v, axis=1)
        # # abr,abr_time = abr_mem_v([stacked_v],duration*1000,ref_v=-65.)
        abs=[]
        for i,non_zero_neuron_times in enumerate(neuron_list):
            abr, abr_time = abr_spikes(non_zero_neuron_times[ear_index], duration * 1000)
            abrs.append(abr*location_weights[i])
        stacked_abrs = np.vstack(tuple(abrs))
        plt.figure("ABR ear{} test {}".format(ear_index, test_index))
        sum_abrs = np.sum(np.asarray(abrs),axis=0)
        plt.plot(abr_time,sum_abrs)
        # for a in abrs:
        #     plt.plot(abr_time,a)
    if plot_stim:
        plt.figure("audio stimulus")
        ax = plt.subplot(len(stimulus), 1, ear_index+1)
        ax.set_title('ear {}'.format(ear_index))
        x = np.linspace(0,duration,len(stimulus[ear_index]))
        ax.plot(x,stimulus[ear_index])

an_count = 0
for ear in an_spikes:
    for neuron in ear:
        an_count+=len(neuron)
print "an spike count = {}".format(an_count)

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