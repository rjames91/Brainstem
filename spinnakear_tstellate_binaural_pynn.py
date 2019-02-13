import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
import time as local_time
from spinnak_ear.spinnakear import SpiNNakEar
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint
from elephant.statistics import isi,cv
#================================================================================================
# Simulation parameters
#================================================================================================
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

sub_pop = False
conn_pre_gen = False
lateral = True

Fs = 100000.#22050.#
dBSPL=60#50
wav_directory = '/home/rjames/SpiNNaker_devel/OME_SpiNN/'
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

freq = 3000
tone_duration = 0.05
tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL-10,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_stereo = np.asarray([tone,tone_r])

noise = generate_signal(signal_type='noise',dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,modulation_depth=1.,fs=Fs,ramp_duration=0.0025,plt=None,silence=True,silence_duration=0.075)

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
                "noise":noise
}
n_fibres = 200
timestep = 0.1#1.0#
required_total_time = 1#50

stimulus_list = ["tone_{}Hz".format(freq)]
duration_dict = {}
test_file = ''
for stim_string in stimulus_list:
    test_file += stim_string + '_'
test_file += str(int(required_total_time))+'s'

# check if any stimuli are in stereo
num_channels=1
for sound_string in stimulus_list:
    sound = sounds_dict[sound_string]
    duration_dict[sound_string]= (len(sound)/Fs)*1000.#ms
    if len(sound.shape)>1:
        num_channels=2
audio_data = [[] for _ in range(num_channels)]
onset_times = [[[]for _ in range(num_channels)]for _ in range(len(stimulus_list))]

chosen_stimulus_list=[]
while 1:
    rand_choice = numpy.random.randint(len(stimulus_list))
    chosen_string = stimulus_list[rand_choice]
    if chosen_string not in chosen_stimulus_list:
        chosen_stimulus_list.append(chosen_string)
    chosen_samples = sounds_dict[chosen_string]
    if len(chosen_samples.shape) > 1:#stereo
        for i,channel in enumerate(chosen_samples):
            onset_found = False
            for sample in channel:
                if not onset_found and abs(sample) >  4e-6:
                    #onset time, duration tuple (both in ms)
                    onset_times[rand_choice][i].append(numpy.round(1000. * (len(audio_data[i]) / Fs)))
                    onset_found = True
                audio_data[i].append(sample)

    else:
        onset_found = False
        for sample in chosen_samples:
            if not onset_found and abs(sample) > 4e-6:
                onset_times[rand_choice][0].append(numpy.round(1000.*(len(audio_data[0])/Fs)))
                onset_found = True
            audio_data[0].append(sample)
            #add zero input to other ear
        if num_channels > 1:
            silence_amp = 1. * 28e-6 * 10. ** (-20. / 20.) # -20dBSPL silence noise
            silence_samples = numpy.random.rand(chosen_samples.size) * silence_amp
            for sample in silence_samples:
                audio_data[1].append(sample)

    if len(audio_data[0]) / Fs > required_total_time:
        break

max_time = 1000. * (len(audio_data[0]) / Fs)

duration = max_time
# duration = 1000.
n_ears = num_channels

input_pops = [[] for _ in range(n_ears)]
t_pops = [[] for _ in range(n_ears)]

an_t_projs = [[] for _ in range(n_ears)]
t_t_projs = [[] for _ in range(n_ears)]

t_spikes = [[] for _ in range(n_ears)]

if conn_pre_gen:
    connection_dicts_file = np.load(input_directory+'/tstellate_{}an_fibres_connectivity.npz'.format
                    (n_fibres))
    connection_dicts = connection_dicts_file['connection_dicts']
else:
    connection_dicts = [{} for _ in range(n_ears)]

#================================================================================================
# SpiNNaker setup
#================================================================================================
time_start = local_time.time()
sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,255)
sim.set_number_of_neurons_per_core(sim.extra_models.Izhikevich_cond,255)

for ear_index in range(n_ears):
    number_of_inputs = n_fibres

    n_total = int(6.66 * number_of_inputs)
    #ratios taken from campagnola & manis 2014 mouse
    n_t = int(n_total * 2./3 * 24./89)

    #================================================================================================
    # Build Populations
    #================================================================================================
    pop_size = max([number_of_inputs,n_t])
    if pop_size != number_of_inputs: # need to scale
        an_spatial=False
    else:
        an_spatial = True
    input_pops[ear_index]=sim.Population(number_of_inputs,SpiNNakEar(audio_input=audio_data[ear_index],fs=Fs,
                                                                     n_channels=number_of_inputs/10,
                                                                     pole_freqs=None),#freq*np.ones(number_of_inputs/10)),
                                         label="AN_Pop_ear{}".format(ear_index))
    t_pops[ear_index]=sim.Population(pop_size,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="t_stellate_fixed_weight_scale_cond{}".format(ear_index))
    t_pops[ear_index].record(["spikes"])

    #================================================================================================
    # AN --> CN Projections
    #================================================================================================
    w2s_t =0.5#0.05#
    n_an_t_connections = 5.
    n_an_t_connections = RandomDistribution('uniform',[4.,6.])
    av_an_t = w2s_t/5.
    # an_t_weight = av_an_t
    an_t_weight = RandomDistribution('uniform',[av_an_t/5.,av_an_t*2])
    if conn_pre_gen:
        an_t_list = connection_dicts[ear_index]['an_t_list']
    else:
        if an_spatial:
            an_t_list = spatial_normal_dist_connection_builder(pop_size,number_of_inputs,n_t,RandomDistribution,conn_num=n_an_t_connections,dist=1.,sigma=1.
                                                    ,conn_weight=an_t_weight)
        else:
            an_t_list = normal_dist_connection_builder(number_of_inputs,n_t,RandomDistribution,conn_num=n_an_t_connections,dist=1.,sigma=1.
                                                    ,conn_weight=an_t_weight)
            #scale up post ids
            scale_factor = float(pop_size-1)/(n_t-1)
            an_t_list = [(an,int(t*scale_factor),w,delay) for (an,t,w,delay) in an_t_list]

    # an_t_list = conn_lists['an_t_list']
    an_t_projs[ear_index] = sim.Projection(input_pops[ear_index],t_pops[ear_index],sim.FromListConnector(an_t_list),synapse_type=sim.StaticSynapse())

    if conn_pre_gen is False:
        connection_dicts[ear_index]['an_t_list']=an_t_list

#now all populations have been created we can create lateral projections
#================================================================================================
# Lateral CN Projections
#================================================================================================
n_lateral_connections = 100.
lateral_connection_strength = w2s_t#*0.1#0.3#0.6
inh_ratio = 0.1#1.
t_lat_sigma = n_total * 0.01

if lateral is True:
    for ear_index in range(n_ears):
        av_t_t = lateral_connection_strength/n_lateral_connections#0.5#0.1#
        t_t_weight = RandomDistribution('normal_clipped',[av_t_t,0.1*av_t_t,0,av_t_t*2.])
        # plt.hist(t_t_weight.next(1000),bins=100)
        if conn_pre_gen:
            t_t_list = connection_dicts[ear_index]['t_t_list']
        else:
            t_t_list = spatial_normal_dist_connection_builder(pop_size,n_t,n_t,RandomDistribution,conn_num=n_lateral_connections,dist=1.,sigma=t_lat_sigma,conn_weight=t_t_weight)
        t_t_projs[ear_index] = sim.Projection(t_pops[ear_index],t_pops[ear_index],sim.FromListConnector(t_t_list),synapse_type=sim.StaticSynapse())

        if conn_pre_gen is False:
            connection_dicts[ear_index]['t_t_list'] = t_t_list

if conn_pre_gen is False:
    np.savez_compressed(input_directory+'/tstellate_{}an_fibres_connectivity'.format
                    (number_of_inputs),connection_dicts=connection_dicts)

max_period = 6000.
num_recordings =1#int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

for ear_index in range(n_ears):
    t_data = t_pops[ear_index].get_data(["all"])
    t_spikes[ear_index] = t_data.segments[0].spiketrains
    # if duration < 6000.:
    #     # psth_plot_8(plt, numpy.arange(175, 225), t_spikes, bin_width=timestep / 1000., duration=duration / 1000.,
    #     #             title="PSTH_T ear{}".format(ear_index))
    #
    #     # for pop in t_pops[ear_index]:
    #     #     data = pop.get_data(["v"])
    #     #     mem_v = data.segments[0].filter(name='v')
    #     #     cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,
    #     #                         title='t stellate pop ear{}'.format(ear_index),id=range(pop.size))

    neuron_title_list = ['t_stellate', 'd_stellate', 'bushy', 'octopus','moc']
    neuron_list = [t_spikes]#, d_spikes, b_spikes, o_spikes,moc_spikes]
    plt.figure("spikes ear{}".format(ear_index))
    for i, neuron_times in enumerate(neuron_list):
        non_zero_neuron_times = neuron_times[ear_index]#[spikes for spikes in neuron_times[ear_index] if len(spikes)>0]#
        spike_raster_plot_8(non_zero_neuron_times, plt, duration/1000., len(non_zero_neuron_times) + 1, 0.001,
                            title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1)
                            )  # ,filepath=results_directory)
        psth_spikes = non_zero_neuron_times[150:200]
        # psth_plot_8(plt, numpy.arange(len(psth_spikes)), psth_spikes, bin_width=0.25 / 1000.,
        #             duration=duration / 1000.,title="PSTH_T ear{}".format(0))

sim.end()
print "simulation of {}s complete in {}s".format(duration/1000.,local_time.time()-time_start)

isi_cut_off = 20
# left_t_spike_trains = [spike_train for spike_train in t_spikes[0] if len(spike_train)>0]
# t_isi = [isi(spike_train) for spike_train in left_t_spike_trains]
t_isi = [isi(spike_train) for spike_train in psth_spikes]
t_isi_filtered = []
for fibre in t_isi:
    t_isi_filtered.append([i for i in fibre if i<isi_cut_off])
# if len(t_isi_filtered)%2 == 0:
#     n_choices = len(t_isi_filtered)
# else:
#     n_choices = len(t_isi_filtered)-1
n_choices = 10
chosen_indices = np.random.choice(len(t_isi_filtered),n_choices,replace=False)
for i,index in enumerate(chosen_indices):
    plt.figure("ISI")
    all_isi = [interval.item() for interval in t_isi_filtered[index]]
    plt.subplot(n_choices/2,2,i+1)
    plt.hist(all_isi)
    plt.xlim((0,isi_cut_off))
    plt.figure("PSTH")
    psth = repeat_test_spikes_gen(psth_spikes,index,onset_times=onset_times[0],test_duration=(1.5*tone_duration)*1000.)[0]
    psth_plot_8(plt, numpy.arange(len(psth)), psth, bin_width=0.25 / 1000., duration=tone_duration*1.5,
            subplots=(n_choices/2,2,i+1))

np.savez_compressed(input_directory+'/tstellate_' + test_file + '_{}an_fibres_{}ms_timestep_{}dB'.format
                     (number_of_inputs,timestep,dBSPL),an_spikes=[],
                     #t_spikes=t_spikes,d_spikes=d_spikes,b_spikes=b_spikes,o_spikes=o_spikes)
                     t_spikes=t_spikes,onset_times=onset_times)

plt.show()