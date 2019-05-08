import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess
import time as local_time
from spinnak_ear.spinnakear import SpiNNakEar,naive_n_chips_calc
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint
from elephant.statistics import isi,cv
#================================================================================================
# Simulation parameters
#================================================================================================

n_tds = 10
t_ds = np.logspace(np.log10(10),np.log10(500),n_tds)
t_stellate_izk_class_2_params = {
               'a':0.4,#0.2,#0.02,#
               'b':0.26,
               'c':-65,
               # 'd':200,
               'u':0,#-15,
               'tau_syn_E': 0.94,#3.0,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -63.0,
}

conn_pre_gen = True
lateral = False
moc_feedback = False
record_en = True
auto_max_atoms = False

Fs = 50e3#100000.#
dBSPL=65#-60#30#
wav_directory = '/home/rjames/SpiNNaker_devel/OME_SpiNN/'
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

freq = 1000
tone_duration = 0.05
silence_duration = 0.025 #0.075#
tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_stereo = np.asarray([tone,tone_r])

timit_l = generate_signal(signal_type='file',dBSPL=dBSPL+20,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=0)
[_,signal] = wavfile.read(wav_directory+'10788_edit.wav')
signal = signal[:,0]
max_val = numpy.max(numpy.abs(signal))
timit_r = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=1,max_val=max_val)
timit = numpy.asarray([timit_l,timit_r])

noise_dur = timit_l.size / Fs
noise = generate_signal(signal_type='noise',dBSPL=dBSPL,duration=noise_dur,
                       modulation_freq=0.,modulation_depth=1.,fs=Fs,ramp_duration=0.0025,plt=None,silence=True,silence_duration=silence_duration)

click = generate_signal(signal_type='click',fs=Fs,dBSPL=dBSPL,duration=0.0002,plt=None,silence=True,silence_duration=0.075)
click_stereo = np.asarray([click,click])

chirp = generate_signal(signal_type='sweep_tone',freq=[30,18e3],fs=Fs,dBSPL=dBSPL,duration=1.-(0.075*2),plt=None,silence=True,silence_duration=0.075)
chirp_stereo = np.asarray([chirp,chirp])

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
                "timit":timit,
                "noise":noise,
                "click":click_stereo,
                "chirp":chirp_stereo
}
n_fibres = 100
timestep = 0.1#1.0#
required_total_time = 0.0002#20#0.1#50.#tone_duration#

stimulus_list = ["tone_{}Hz".format(freq)]#['chirp']#['timit']#["click"]#
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

duration = max_time * 1.1
# duration = 1000.
n_ears = num_channels

input_pops = [[] for _ in range(n_ears)]
spinnakear_objects = [[] for _ in range(n_ears)]
t_pops = [[[] for __ in range(n_tds)] for _ in range(n_ears)]


an_t_projs = [[] for _ in range(n_ears)]


t_spikes = [[[] for __ in range(n_tds)] for _ in range(n_ears)]
an_spikes = [[] for _ in range(n_ears)]
moc_att = [[] for _ in range(n_ears)]


if conn_pre_gen:
    try:
        connection_dicts_file = np.load(input_directory+'/cn_{}an_fibres_{}ears_connectivity.npz'.format
                        (n_fibres,n_ears))
        connection_dicts = connection_dicts_file['connection_dicts']
    except:
        connection_dicts = [{} for _ in range(n_ears)]
        conn_pre_gen=False
else:
    connection_dicts = [{} for _ in range(n_ears)]

#================================================================================================
# SpiNNaker setup
#================================================================================================
# n_total = int(6.66 * n_fibres)
n_total = int(2.4 * n_fibres)
#ratios taken from campagnola & manis 2014 mouse
n_t = int(n_total * 2./3 * 24./89)
n_d = int(n_total * 1./3 * 24./89)
n_b = int(n_total * 55./89)#number_of_inputs#
n_o = int(n_total * 10./89.)
# n_moc = int(n_fibres * (360/30e3))
# n_moc = int(n_fibres)
n_moc = 360
n_sub_t=int(n_t/n_tds)

max_neurons_per_core = 255
n_chips_required = naive_n_chips_calc(n_fibres/10,n_ears,[(n_t,max_neurons_per_core),(n_d,max_neurons_per_core),
                                                          (n_b,max_neurons_per_core),(n_o,max_neurons_per_core),
                                                          (n_moc,max_neurons_per_core)])
time_start = local_time.time()
sim.setup(timestep=timestep,n_chips_required=n_chips_required)

sim.set_number_of_neurons_per_core(sim.IF_cond_exp,max_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.extra_models.Izhikevich_cond,max_neurons_per_core)

for ear_index in range(n_ears):
    number_of_inputs = n_fibres

    spinnakear_param_file = input_directory+'/spinnakear_params_ear{}_{}fibres.npz'.format(ear_index,n_fibres)
    spinnakear_objects[ear_index] = SpiNNakEar(audio_input=audio_data[ear_index],fs=Fs,
                                                                     n_channels=number_of_inputs/10,
                                                                     pole_freqs=None,param_file=spinnakear_param_file,ear_index=ear_index)#freq*np.ones(number_of_inputs/10))
    input_pops[ear_index]=sim.Population(number_of_inputs,spinnakear_objects[ear_index],label="AN_Pop_ear{}".format(ear_index))
    input_pops[ear_index].record(['spikes','moc'])

    #================================================================================================
    # Build Populations
    #================================================================================================
    pop_size = max([number_of_inputs,n_d,n_t,n_b,n_o,n_moc])
    for td in range(n_tds):
        t_stellate_izk_class_2_params['d']=t_ds[td]
        t_pops[ear_index][td] = sim.Population(n_sub_t, sim.extra_models.Izhikevich_cond, t_stellate_izk_class_2_params,
                                           label="t_stellate_fixed_weight_scale_cond{}".format(ear_index))
    if 1:#record_en is True:
        for pop in t_pops[ear_index]:
            pop.record(["spikes"])

    #================================================================================================
    # AN --> CN Projections
    #================================================================================================
    w2s_t = 1.6#0.8#0.5#0.25#0.1#0.3#0.1#0.7
    n_an_t_connections = RandomDistribution('uniform',[4.,6.])
    av_an_t = w2s_t/5.
    # an_t_weight = RandomDistribution('uniform',[0,av_an_t*2])
    # an_t_weight = RandomDistribution('uniform',[av_an_t/5.,av_an_t*2])
    an_t_weight = RandomDistribution('normal_clipped',[av_an_t,0.1*av_an_t,0,av_an_t*2.])
    if conn_pre_gen:
        an_t_list = connection_dicts[ear_index]['an_t_list']
    else:
        an_t_master,max_dist = normal_dist_connection_builder(number_of_inputs,n_t,RandomDistribution,conn_num=n_an_t_connections,
                                                dist=1.,sigma=1.,conn_weight=an_t_weight,normalised_space=pop_size,get_max_dist=True)
        an_t_list = [[]for _ in range(n_tds)]
        for (pre,post,w,d) in an_t_master:
            i = np.remainder(post, n_tds)
            an_t_list[i].append((pre, int((float(post)/n_t)*n_sub_t + 0.5), w, d))

    for i,source_l in enumerate(an_t_list):
        if len(source_l)>0:
            an_t_projs[ear_index].append(sim.Projection(input_pops[ear_index],t_pops[ear_index][i],sim.FromListConnector(source_l),synapse_type=sim.StaticSynapse()))

max_period = 6000.
num_recordings =1#int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

if record_en is True:
    for ear_index in range(n_ears):
        for i,pop in enumerate(t_pops[ear_index]):
            t_data = pop.get_data(["spikes"])
            t_spikes[ear_index][i] = t_data.segments[0].spiketrains
        ear_data = input_pops[ear_index].get_data(['spikes','moc'])
        an_spikes[ear_index] = ear_data['spikes']
        moc_att[ear_index] = ear_data['moc']
sim.end()
print "simulation of {}s complete in {}s".format(duration/1000.,local_time.time()-time_start)


if record_en:
    np.savez_compressed(input_directory+'/t_stellate_' + test_file + '_{}an_fibres_{}ms_timestep_{}dB_{}s_moc_{}_lat_{}'.format
                         (number_of_inputs,timestep,dBSPL,int(duration/1000.),moc_feedback,lateral),an_spikes=an_spikes,
                         t_spikes=t_spikes,onset_times=onset_times,
                        moc_att=moc_att,Fs=Fs)

    plt.show()