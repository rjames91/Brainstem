import spynnaker8 as sim
import numpy as np
import pylab as plt
import sys
sys.path.append("../PyNN8Examples")
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
octopus_params_cond_izh = {
                'a':0.02,
                'b':0.1,#0.25,#
                'c':-65,
                'd':2,
                'u':-5,
               'tau_syn_E': 0.2,#0.35,#2.5,#
               'e_rev_E': 30.,#-10.,#-55.,#-35.,#-55.1,#
               'v': -70.,
               }

bushy_params_cond = {#'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 1.,#10.0,#2.,#3.,#
               'tau_syn_E': 0.15,#2.5,#
               }

n_tds = 10
# t_ds = np.logspace(np.log10(10),np.log10(500),n_tds)
t_ds = np.logspace(np.log10(1),np.log10(150),n_tds)
t_stellate_izk_class_2_params = {
               'a':0.4,#0.2,#0.02,#
               'b':0.26,
               'c':-65,
               # 'd':200,
               'u':0,#-15,
               'tau_syn_E': 4.0,#0.94,#3.0,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -63.0,
}
n_dds = 10
d_ds = np.logspace(np.log10(10),np.log10(50),n_dds)
d_ds = np.linspace(10,50,n_dds)
d_stellate_izk_class_1_params = {
               'a':0.05,#0.02,
               'b':-0.1,
               'c':-55,
               'd':4,
               'u':10,
               'tau_syn_E':4.88,
               'e_rev_E': 0.,
               'tau_syn_I':4.,
               'v': -63.0,
}
moc_lts_params = {
    'a': 0.02,
    'b': 0.25,
    'c': -65,
    'd':2,
    'u': -10,
    'v': -65,
}
conn_pre_gen = False
lateral = True
moc_feedback = True
record_en = True
record_vars = ['spikes']
auto_max_atoms = False

Fs = 50e3#100000.#
dBSPL=65#-60#30#
wav_directory = '../OME_SpiNN/'
input_directory = '/tmp/rob_test_results/'

freq = 1000
mod_freq = 100.
tone_duration = 0.1
silence_duration = 0.2#0.025 #0.075#
tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_stereo = np.asarray([tone,tone_r])

tone = generate_signal(freq=freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=mod_freq,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
tone_r = generate_signal(freq=mod_freq,dBSPL=dBSPL,duration=tone_duration,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=silence_duration)
sam_tone_stereo = np.asarray([tone,tone_r])


timit_l = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,silence_duration=silence_duration,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=0)
[_,signal] = wavfile.read(wav_directory+'10788_edit.wav')
signal = signal[:,0]
max_val=numpy.mean(signal**2)**0.5
timit_r = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,silence_duration=silence_duration,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=1,max_val=max_val)
timit = numpy.asarray([timit_l,timit_r])

noise_dur = timit_l.size / Fs
noise = generate_signal(signal_type='noise',dBSPL=dBSPL,duration=noise_dur,
                       modulation_freq=0.,modulation_depth=1.,fs=Fs,ramp_duration=0.0025,plt=None,silence=True,silence_duration=silence_duration)

click = generate_signal(signal_type='click',fs=Fs,dBSPL=dBSPL,duration=0.001,ramp_duration=0.0001,plt=None,silence=True,silence_duration=silence_duration)
click_stereo = np.asarray([click,click])

chirp = generate_signal(signal_type='sweep_tone',freq=[30,18e3],fs=Fs,dBSPL=dBSPL,duration=1.-(0.075*2),plt=None,silence=True,silence_duration=0.075)
chirp_stereo = np.asarray([chirp,chirp])

sounds_dict = {
                "tone_{}Hz".format(freq):tone,
                "tone_{}Hz_stereo".format(freq):tone_stereo,
                "timit":timit,
                "noise":noise,
                "click":click_stereo,
                "chirp":chirp_stereo,
                "{}Hz_sam_tone{}Hz".format(mod_freq,freq):sam_tone_stereo
}
n_fibres = 1000
timestep = 0.1#1.0#
required_total_time = 0.0002#0.2#20#20#0.1#50.#tone_duration#

stimulus_list = ['timit']#["{}Hz_sam_tone{}Hz".format(mod_freq,freq)]#["tone_{}Hz_stereo".format(freq)]#["click"]#['chirp']#
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
print "simulation real time = {}s".format(duration*0.001)
# duration = 1000.
n_ears = num_channels

input_pops = [[] for _ in range(n_ears)]
spinnakear_objects = [[] for _ in range(n_ears)]
sg_pops = [[] for _ in range(n_ears)]
t_pops = [[[] for __ in range(n_tds)] for _ in range(n_ears)]
d_pops = [[[] for __ in range(n_dds)] for _ in range(n_ears)]
b_pops = [[] for _ in range(n_ears)]
o_pops = [[] for _ in range(n_ears)]
moc_pops = [[] for _ in range(n_ears)]

ear_an_projs = [[] for _ in range(n_ears)]
an_t_projs = [[] for _ in range(n_ears)]
an_d_projs = [[] for _ in range(n_ears)]
an_b_projs = [[] for _ in range(n_ears)]
an_o_projs = [[] for _ in range(n_ears)]
t_d_projs = [[] for _ in range(n_ears)]
t_t_projs = [[] for _ in range(n_ears)]
t_b_projs = [[] for _ in range(n_ears)]
t_mocc_projs = [[] for _ in range(n_ears)]
d_t_projs = [[] for _ in range(n_ears)]
d_d_projs = [[] for _ in range(n_ears)]
d_b_projs = [[] for _ in range(n_ears)]
d_tc_projs = [[] for _ in range(n_ears)]
d_dc_projs = [[] for _ in range(n_ears)]
moc_anc_projs = [[] for _ in range(n_ears)]
moc_an_projs = [[] for _ in range(n_ears)]

t_incoming = [[] for _ in range(n_ears)]
d_incoming = [[] for _ in range(n_ears)]
b_incoming = [[] for _ in range(n_ears)]
o_incoming = [[] for _ in range(n_ears)]
moc_incoming = [[] for _ in range(n_ears)]

sg_data = [[] for _ in range(n_ears)]
t_data = [[[] for __ in range(n_tds)] for _ in range(n_ears)]
d_data = [[[] for __ in range(n_dds)] for _ in range(n_ears)]
b_data = [[] for _ in range(n_ears)]
o_data = [[] for _ in range(n_ears)]
moc_data = [[] for _ in range(n_ears)]
ear_data = [[] for _ in range(n_ears)]
# an_spikes = [[] for _ in range(n_ears)]
# moc_att = [[] for _ in range(n_ears)]

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
n_sub_d = int(n_d/n_dds)

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
                                                                     pole_freqs=None,param_file=spinnakear_param_file,ear_index=ear_index,duration=duration)#freq*np.ones(number_of_inputs/10))
    input_pops[ear_index]=sim.Population(number_of_inputs,spinnakear_objects[ear_index],label="AN_Pop_ear{}".format(ear_index))
    input_pops[ear_index].record(['spikes','moc'])
    sg_pops[ear_index]=sim.Population(number_of_inputs,sim.IF_cond_exp,{'tau_m':5.,'tau_syn_E':0.1},label="SG_Pop_ear_fixed_weight_scale_cond{}".format(ear_index))

    #================================================================================================
    # Build Populations
    #================================================================================================
    pop_size = max([number_of_inputs,n_d,n_t,n_b,n_o,n_moc])
    # plt.figure("stimulus ear {}".format(ear_index))
    # plt.plot(audio_data[ear_index])
    for dd in range(n_dds):
        d_stellate_izk_class_1_params['d']=d_ds[dd]
        d_pops[ear_index][dd]=sim.Population(n_sub_d,sim.extra_models.Izhikevich_cond,d_stellate_izk_class_1_params,
                                             label="d_stellate_fixed_weight_scale_cond{}".format(ear_index))

    for td in range(n_tds):
        t_stellate_izk_class_2_params['d']=t_ds[td]
        t_pops[ear_index][td] = sim.Population(n_sub_t, sim.Izhikevich, t_stellate_izk_class_2_params,
                                           label="t_stellate_fixed_weight_scale{}".format(ear_index))
    #spherical bushy
    b_pops[ear_index]=sim.Population(n_b,sim.IF_cond_exp,bushy_params_cond,label="bushy_fixed_weight_scale_cond{}".format(ear_index))

    o_pops[ear_index]=sim.Population(n_o,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="octopus_fixed_weight_scale_cond{}".format(ear_index))
    moc_pops[ear_index] = sim.Population(n_moc, sim.Izhikevich, moc_lts_params, label="moc{}".format(ear_index))
    # moc_pops[ear_index] = sim.Population(n_moc, sim.IF_cond_exp, moc_params, label="moc_fixed_weight_scale_cond{}".format(ear_index))

    sg_pops[ear_index].record(record_vars)
    b_pops[ear_index].record(record_vars)
    for pop in t_pops[ear_index]:
        pop.record(record_vars)
    o_pops[ear_index].record(record_vars)
    moc_pops[ear_index].record(record_vars)
    for pop in d_pops[ear_index]:
        pop.record(record_vars)


    #================================================================================================
    # AN --> CN Projections
    #================================================================================================
    w2s_sg = 3.
    ear_an_list = [(i,i,w2s_sg,timestep) for i in range(number_of_inputs)]
    ear_an_projs[ear_index]=sim.Projection(input_pops[ear_index],sg_pops[ear_index],sim.FromListConnector(ear_an_list),
                                           synapse_type=sim.StaticSynapse())
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
                                                dist=1.,sigma=1.,conn_weight=an_t_weight,delay=timestep,normalised_space=pop_size,get_max_dist=True)
        t_incoming[ear_index].append(max_dist*(float(number_of_inputs)/pop_size))#weighted by how many pre neurons reside in pop_size 'space'
        an_t_list = [[]for _ in range(n_tds)]
        for (pre,post,w,d) in an_t_master:
            i = np.remainder(post, n_tds)
            an_t_list[i].append((pre, int((float(post)/n_t)*n_sub_t + 0.5), w, d))

    for i,source_l in enumerate(an_t_list):
        if len(source_l)>0:
            an_t_projs[ear_index].append(sim.Projection(sg_pops[ear_index],t_pops[ear_index][i],sim.FromListConnector(source_l),synapse_type=sim.StaticSynapse()))

    # n_an_d_connections = RandomDistribution('uniform',[11.,88.])
    n_an_d_connections = RandomDistribution('normal_clipped',[60.,5.,11.,88.])#estimation of upper and lower bounds Manis 2017
    w2s_d = 1.#0.75#1.5#0.3#1.#0.5#
    av_an_d = w2s_d/88.#w2s_d/60.#
    # an_d_weight = RandomDistribution('uniform',[0,av_an_d])
    an_d_weight = RandomDistribution('normal_clipped',[av_an_d,0.1*av_an_d,0,av_an_d*2.])
    if conn_pre_gen:
        an_d_list = connection_dicts[ear_index]['an_d_list']
    else:
        an_d_master,max_dist = normal_dist_connection_builder(number_of_inputs,n_d,RandomDistribution,conn_num=n_an_d_connections,dist=1.,
                                           sigma=pop_size/15.,conn_weight=an_d_weight,delay=timestep,normalised_space=pop_size,get_max_dist=True)
        d_incoming[ear_index].append(max_dist*(float(number_of_inputs)/pop_size))
        an_d_list = [[] for _ in range(n_dds)]
        for (pre, post, w, d) in an_d_master:
            i = np.remainder(post, n_dds)
            an_d_list[i].append((pre, int((float(post)/n_d)*n_sub_d + 0.5), w, d))

    for i, source_l in enumerate(an_d_list):
        if len(source_l)>0:
            an_d_projs[ear_index].append(
            sim.Projection(sg_pops[ear_index], d_pops[ear_index][i], sim.FromListConnector(source_l),
                           synapse_type=sim.StaticSynapse()))

    n_an_b_connections = RandomDistribution('uniform',[2.,5.])
    w2s_b = 5.#0.3#
    av_an_b = w2s_b/5.
    an_b_weight = RandomDistribution('normal_clipped',[av_an_b,0.1*av_an_b,0,av_an_b*2.])
    if conn_pre_gen:
        an_b_list = connection_dicts[ear_index]['an_b_list']
    else:
        an_b_list,max_dist = normal_dist_connection_builder(number_of_inputs,n_b,RandomDistribution,conn_num=n_an_b_connections,dist=1.,
                                           sigma=1.,conn_weight=an_b_weight,delay=1.,normalised_space=pop_size,get_max_dist=True)
        b_incoming[ear_index].append(max_dist*(float(number_of_inputs)/pop_size))

    an_b_projs[ear_index] = sim.Projection(sg_pops[ear_index], b_pops[ear_index], sim.FromListConnector(an_b_list),
                                       synapse_type=sim.StaticSynapse())
    w2s_o = 1.5#3.#2.#7.
    n_an_o_connections = RandomDistribution('uniform',[30.,120.])
    # n_an_o_connections = RandomDistribution('normal_clipped',[50.,5.,30.,120.])
    av_an_o = w2s_o/50.
    an_o_weight = RandomDistribution('normal_clipped', [av_an_o, 0.1 * av_an_o, 0, av_an_o * 2.])
    if conn_pre_gen:
        an_o_list = connection_dicts[ear_index]['an_o_list']
    else:
        an_o_list,max_dist = normal_dist_connection_builder(number_of_inputs, n_o, RandomDistribution,
                                                   conn_num=n_an_o_connections, dist=1.,
                                                   sigma=pop_size/20., conn_weight=an_o_weight,delay=1.,
                                                   normalised_space=pop_size,get_max_dist=True)
        o_incoming[ear_index].append(max_dist*(float(number_of_inputs)/pop_size))
    an_o_projs[ear_index] = sim.Projection(sg_pops[ear_index], o_pops[ear_index], sim.FromListConnector(an_o_list),
                                           synapse_type=sim.StaticSynapse())

    if conn_pre_gen is False:
        connection_dicts[ear_index]['an_t_list']=an_t_list
        connection_dicts[ear_index]['an_d_list']=an_d_list
        connection_dicts[ear_index]['an_o_list']=an_o_list
        connection_dicts[ear_index]['an_b_list']=an_b_list

#once all populations have been created we can create lateral projections
#================================================================================================
# Lateral CN Projections
#================================================================================================
max_n_lat_conn = 50.
lat_conn_strength = 0.1#1.#0.6#
lateral_connection_weight = lat_conn_strength/max_n_lat_conn

#uniform distribution of n lateral synapses per cell
#dependent on span of dendrites and size of pre population
stellate_dendrite_size = 1.
bushy_dendrite_size = 0.5
n_stellate = n_d + n_t
t_ratio = float(n_t)/n_stellate
d_ratio = (float(n_d)/n_stellate)*0.5

t_t_n_conn = RandomDistribution('uniform',[0,max_n_lat_conn*stellate_dendrite_size*t_ratio])
t_d_n_conn = RandomDistribution('uniform',[0,max_n_lat_conn*stellate_dendrite_size*t_ratio])
d_t_n_conn = RandomDistribution('uniform',[0,max_n_lat_conn*stellate_dendrite_size*d_ratio])
d_b_n_conn = RandomDistribution('uniform',[0,max_n_lat_conn*bushy_dendrite_size*d_ratio])

t_lat_sigma = math.sqrt(pop_size * 0.1)
d_lat_sigma = math.sqrt(pop_size * 0.1)
b_lat_sigma = math.sqrt(pop_size * 0.01)

for ear_index in range(n_ears):
    if lateral is True:
        av_lat_t = lateral_connection_weight * av_an_t
        lat_t_weight = RandomDistribution('normal_clipped',[av_lat_t,0.1*av_lat_t,0,av_lat_t*2.])
        av_lat_d = lateral_connection_weight * av_an_d
        lat_d_weight = RandomDistribution('normal_clipped',[av_lat_d,0.1*av_lat_d,0,av_lat_d*2.])
        av_lat_b = lateral_connection_weight * av_an_b
        lat_b_weight = RandomDistribution('normal_clipped',[av_lat_b,0.1*av_lat_b,0,av_lat_b*2.])

        if conn_pre_gen:
            t_t_list = connection_dicts[ear_index]['t_t_list']
        else:
            t_t_master,max_dist=normal_dist_connection_builder(n_t,n_t,RandomDistribution,conn_num=t_t_n_conn,dist=1.,
                                                               sigma=t_lat_sigma,conn_weight=lat_t_weight,normalised_space=pop_size,
                                                               get_max_dist=True)
            t_incoming[ear_index].append(max_dist*(float(n_t)/pop_size))
            t_t_list = [[[] for _ in range(n_tds)] for __ in range(n_tds)]
            for (pre,post,w,d) in t_t_master:
                i = np.remainder(pre,n_tds)
                j = np.remainder(post,n_tds)
                t_t_list[i][j].append((int((float(pre) / n_t) * n_sub_t + 0.5),int((float(post) / n_t) * n_sub_t + 0.5),w,d))

        for i,source_l in enumerate(t_t_list):
            for j,target_l in enumerate(source_l):
                if len(target_l) > 0:
                    t_t_projs[ear_index].append(sim.Projection(t_pops[ear_index][i],t_pops[ear_index][j],sim.FromListConnector(target_l),synapse_type=sim.StaticSynapse()))

        if conn_pre_gen:
            t_d_list = connection_dicts[ear_index]['t_d_list']
        else:
            t_d_master,max_dist= normal_dist_connection_builder(n_t, n_d, RandomDistribution, conn_num=t_d_n_conn, dist=1.,
                                               sigma=d_lat_sigma, conn_weight=lat_d_weight, normalised_space=pop_size,get_max_dist=True)
            d_incoming[ear_index].append(max_dist*(float(n_t)/pop_size))
            t_d_list = [[[] for _ in range(n_dds)] for __ in range(n_tds)]
            for (pre,post,w,d) in t_d_master:
                i = np.remainder(pre,n_tds)
                j = np.remainder(post,n_dds)
                int((float(post) / n_t) * n_sub_t + 0.5)
                t_d_list[i][j].append((int((float(pre) / n_t) * n_sub_t + 0.5),int((float(post) / n_d) * n_sub_d + 0.5),w,d))

        for i,source_l in enumerate(t_d_list):
            for j, target_l in enumerate(source_l):
                if len(target_l)>0:
                    t_d_projs[ear_index] = sim.Projection(t_pops[ear_index][i],d_pops[ear_index][j],sim.FromListConnector(target_l),synapse_type=sim.StaticSynapse())

        if conn_pre_gen:
            d_t_list = connection_dicts[ear_index]['d_t_list']
        else:
            d_t_master,max_dist = normal_dist_connection_builder(n_d,n_t,RandomDistribution,conn_num=d_t_n_conn,dist=1.,
                                                                 sigma=t_lat_sigma,conn_weight=lat_t_weight,
                                                                 normalised_space=pop_size,get_max_dist=True)
            t_incoming[ear_index].append(max_dist*(float(n_d)/pop_size))
            d_t_list = [[[] for _ in range(n_tds)] for __ in range(n_dds)]
            for (pre, post, w, d) in d_t_master:
                i = np.remainder(pre, n_dds)
                j = np.remainder(post,n_tds)
                d_t_list[i][j].append((int((float(pre) / n_d) * n_sub_d + 0.5), int((float(post) / n_t) * n_sub_t + 0.5), w, d))

        for i, source_l in enumerate(d_t_list):
            for j, target_l in enumerate(source_l):
                if len(target_l)>0:
                    d_t_projs[ear_index] = sim.Projection(d_pops[ear_index][i], t_pops[ear_index][j],
                                                  sim.FromListConnector(target_l), synapse_type=sim.StaticSynapse(),
                                                  receptor_type='inhibitory')

        if conn_pre_gen:
            d_b_list = connection_dicts[ear_index]['d_b_list']
        else:
            d_b_master,max_dist = normal_dist_connection_builder(n_d,n_b,RandomDistribution,conn_num=d_b_n_conn,dist=1.,
                                                      sigma=b_lat_sigma,conn_weight=lat_b_weight,
                                                      normalised_space=pop_size,get_max_dist=True)
            b_incoming[ear_index].append(max_dist*(float(n_d)/pop_size))
            d_b_list = [[] for _ in range(n_dds)]
            for (pre,post,w,d) in d_b_master:
                i = np.remainder(pre, n_dds)
                d_b_list[i].append((int((float(pre) / n_d) * n_sub_d + 0.5),post, w, d))
        for i, source_l in enumerate(d_b_list):
            if len(source_l)>0:
                d_b_projs[ear_index] = sim.Projection(d_pops[ear_index][i],b_pops[ear_index],sim.FromListConnector(source_l),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')

        if conn_pre_gen is False:
            connection_dicts[ear_index]['t_t_list'] = t_t_list
            connection_dicts[ear_index]['t_d_list'] = t_d_list
            connection_dicts[ear_index]['d_t_list'] = d_t_list
            connection_dicts[ear_index]['d_b_list'] = d_b_list
    #TODO: verify contralateral d->tc
    if n_ears>1:
        contra_ear_index = n_ears - 1 - ear_index
        if lateral is True:
            if conn_pre_gen:
                d_tc_list = connection_dicts[ear_index]['d_tc_list']
            else:
                d_tc_master,max_dist = normal_dist_connection_builder(n_d,n_t,RandomDistribution,conn_num=d_t_n_conn,dist=1.,
                                                             sigma=t_lat_sigma,conn_weight=lat_t_weight,
                                                             normalised_space=pop_size,get_max_dist=True)
                t_incoming[contra_ear_index].append(max_dist*(float(n_d)/pop_size))
                d_tc_list = [[[] for _ in range(n_tds)] for __ in range(n_dds)]
                for (pre,post,w,d) in d_tc_master:
                    i = np.remainder(pre, n_dds)
                    j= np.remainder(post, n_tds)
                    d_tc_list[i][j].append((int((float(pre) / n_d) * n_sub_d + 0.5), int((float(post) / n_t) * n_sub_t + 0.5), w, d))

                connection_dicts[ear_index]['d_tc_list'] = d_tc_list

            for i, source_l in enumerate(d_tc_list):
                for j, target_l in enumerate(source_l):
                    if len(target_l)>0:
                        d_tc_projs[ear_index] = sim.Projection(d_pops[ear_index][i], t_pops[contra_ear_index][j],
                                                       sim.FromListConnector(target_l),
                                                       synapse_type=sim.StaticSynapse(), receptor_type='inhibitory')
        # ================================================================================================
        # CN --> VNTB Projections
        # ================================================================================================
        w2s_moc = 10.#0.2#0.6#0.15#0.1#0.05#0.75
        # n_t_moc_connections = RandomDistribution('uniform', [5, 10])
        av_t_moc_connections = 10#int(np.ceil(float(n_t)/n_moc))
        n_t_moc_connections = RandomDistribution('normal_clipped',
                                                  [av_t_moc_connections, 0.1 * av_t_moc_connections, 0,
                                                   av_t_moc_connections * 2.])
        av_t_moc = w2s_moc / av_t_moc_connections#9.
        t_moc_weight = RandomDistribution('normal_clipped', [av_t_moc, 0.1 * av_t_moc, 0, av_t_moc * 2.])
        # t_moc_weight = RandomDistribution('uniform',[av_t_moc/5.,av_t_moc*2])
        if conn_pre_gen:
            t_mocc_list = connection_dicts[ear_index]['t_mocc_list']
        else:
            t_mocc_master,max_dist = normal_dist_connection_builder(n_t, n_moc, RandomDistribution,
                                                             conn_num=n_t_moc_connections, dist=1.,sigma= float(pop_size)/n_moc,
                                                             conn_weight=t_moc_weight,get_max_dist=True,normalised_space=pop_size)
            moc_incoming[contra_ear_index].append(max_dist*(float(n_t)/pop_size))
            t_mocc_list = [[] for _ in range(n_tds)]
            for (pre,post,w,d) in t_mocc_master:
                i = np.remainder(pre, n_tds)
                t_mocc_list[i].append((int((float(pre) / n_t) * n_sub_t + 0.5),post, w, d))

            connection_dicts[ear_index]['t_mocc_list'] = t_mocc_list
        for i,t_mocc_l in enumerate(t_mocc_list):
            if len(t_mocc_l)>0:
                t_mocc_projs[ear_index] = sim.Projection(t_pops[ear_index][i], moc_pops[contra_ear_index],
                                                 sim.FromListConnector(t_mocc_l),synapse_type=sim.StaticSynapse())
        #ipsilateral moc
        n_ohcs = int(number_of_inputs/10.)
        av_n_moc_connections = 5#10#int(np.ceil(float(n_ohcs)/n_moc))
        n_moc_an_connections = RandomDistribution('normal_clipped', [av_n_moc_connections, 0.1 * av_n_moc_connections, 0, av_n_moc_connections * 2.])
        # n_moc_an_connections = 5
        # n_moc_an_connections = RandomDistribution('uniform', [5, 10])
        moc_an_weight = 1.
        uncrossed_sigma = math.sqrt(n_moc/10.)#1 octave of full range
        if conn_pre_gen:
            moc_an_list = connection_dicts[ear_index]['moc_an_list']
        else:
            moc_an_list = normal_dist_connection_builder(n_moc,n_ohcs, RandomDistribution,
                                                                 conn_num=n_moc_an_connections, dist=1.,
                                                                 sigma=uncrossed_sigma, conn_weight=moc_an_weight,multapses=False)

            connection_dicts[ear_index]['moc_an_list'] = moc_an_list
        if moc_feedback:
            moc_an_projs[ear_index] = sim.Projection(moc_pops[ear_index], input_pops[ear_index],
                                                 sim.FromListConnector(moc_an_list),
                                                 synapse_type=sim.StaticSynapse())
        #contralateral moc
        if conn_pre_gen:
            moc_anc_list = connection_dicts[ear_index]['moc_anc_list']
        else:
            moc_anc_list = normal_dist_connection_builder(n_moc,n_ohcs, RandomDistribution,
                                                                 conn_num=n_moc_an_connections, dist=1.,
                                                                 sigma=uncrossed_sigma/2., conn_weight=moc_an_weight,multapses=False)
            connection_dicts[ear_index]['moc_anc_list'] = moc_anc_list
        if moc_feedback:
            moc_anc_projs[ear_index] = sim.Projection(moc_pops[ear_index], input_pops[contra_ear_index],
                                                 sim.FromListConnector(moc_anc_list),
                                                 synapse_type=sim.StaticSynapse())


if conn_pre_gen is False:
    #todo: save all the calculated population max atom constraints
    max_atoms = {}
    target_callback_time_ms = 1.
    # d_slope =0.000234
    # d_int = 0.812474
    # max_est = np.sum(d_incoming,axis=1)*d_slope + d_int
    # max_atoms['d'] = max_neurons_per_core * (target_callback_time_ms/max_est)
    # t_slope =0.000282
    # t_int = 0.829516
    # max_est = np.sum(t_incoming,axis=1)*t_slope + t_int
    # max_atoms['t'] = max_neurons_per_core * (target_callback_time_ms/max_est)
    # b_slope =0.000390
    # b_int = 0.343330
    # max_est = np.sum(b_incoming,axis=1)*b_slope + b_int
    # max_atoms['b'] = max_neurons_per_core * (target_callback_time_ms/max_est)
    # o_slope =0.000397
    # o_int = 0.311144
    # max_est = np.sum(o_incoming,axis=1)*o_slope + o_int
    # max_atoms['o'] = max_neurons_per_core * (target_callback_time_ms/max_est)
    # moc_slope = 0.000077
    # moc_int = -0.102860
    # max_est = np.sum(moc_incoming,axis=1)*moc_slope + moc_int
    # max_atoms['moc'] =max_neurons_per_core * (target_callback_time_ms/max_est)
   # # max_atoms['moc'] = np.asarray([max_neurons_per_core,max_neurons_per_core])

    max_atoms['d'] = d_incoming
    max_atoms['t'] = t_incoming
    max_atoms['b'] = b_incoming
    max_atoms['o'] = o_incoming
    max_atoms['moc'] = moc_incoming

    np.savez_compressed(input_directory + '/cn_{}an_fibres_{}ears_connectivity.npz'.format
    (n_fibres, n_ears),connection_dicts=connection_dicts,max_atoms=max_atoms)

else:
    max_atoms = connection_dicts_file['max_atoms'].item()

pop_list = ['d','t','b','o','moc']
pop_dict = {
        'd':d_pops,
        't':t_pops,
        'b':b_pops,
        'o':o_pops,
        'moc':moc_pops
}
if auto_max_atoms:
    print max_atoms
    for pop in pop_list:
        for ear_index, max_a in enumerate(max_atoms[pop]):

            if max_a>max_neurons_per_core:
                max_a = max_neurons_per_core
            if pop == 't':
                for i in range(n_tds):
                    pop_dict[pop][ear_index][i].set_constraint(MaxVertexAtomsConstraint(max_a))
            elif pop == 'd':
                for i in range(n_dds):
                    pop_dict[pop][ear_index][i].set_constraint(MaxVertexAtomsConstraint(max_a))
            else:
                pop_dict[pop][ear_index].set_constraint(MaxVertexAtomsConstraint(max_a))


# max_atoms = connection_dicts_file['max_atoms'].item()
# o_incoming = max_atoms['o']
# target_callback_time_ms = 0.7
# o_slope =0.000290
# o_int = 0.727078
# max_est = np.max(o_incoming,axis=1).max()*o_slope + o_int
# print "max est = {}".format(max_est)
# max_atoms_o = max_neurons_per_core * (target_callback_time_ms/max_est)
# print "max atoms o = {}".format(max_atoms_o)
#
#
# for i,pop in enumerate(o_pops):
#     max_a = max_atoms_o#[i]
#     if max_a>255:
#         max_a=255
#     pop.set_constraint(MaxVertexAtomsConstraint(max_a))

max_period = 6000.
num_recordings =1#int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

if record_en is True:
    for ear_index in range(n_ears):
        sg_data[ear_index]= sg_pops[ear_index].get_data(record_vars)
        for i,pop in enumerate(t_pops[ear_index]):
            t_data[ear_index][i] = pop.get_data(record_vars)
            # t_spikes[ear_index][i] = t_data.segments[0].spiketrains
        # mem_v = t_data.segments[0].filter(name='v')
        # cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=timestep/1000.,
        #                     title='t stellate pop ear{}'.format(ear_index),id=0)
        for i,pop in enumerate(d_pops[ear_index]):
            d_data[ear_index][i] = pop.get_data(record_vars)
            # d_spikes[ear_index][i] = d_data.segments[0].spiketrains
        # mem_v = d_data.segments[0].filter(name='v')
        # cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,
        #                     title='d stellate pop ear{}'.format(ear_index),id=range(n_d))
        b_data[ear_index] = b_pops[ear_index].get_data(record_vars)
        # b_spikes[ear_index] = b_data.segments[0].spiketrains
        o_data[ear_index] = o_pops[ear_index].get_data(record_vars)
        # o_spikes[ear_index] = o_data.segments[0].spiketrains
        moc_data[ear_index] = moc_pops[ear_index].get_data(record_vars)
        # moc_spikes[ear_index] = moc_data.segments[0].spiketrains

        # ear_data = input_pops[ear_index].get_data(['spikes'])
        ear_data[ear_index] = input_pops[ear_index].get_data(['spikes','moc'])
        # an_spikes[ear_index] = ear_data['spikes']
        # moc_att[ear_index] = ear_data['moc']

        # neuron_title_list = ['t_stellate', 'd_stellate', 'bushy', 'octopus','moc','an']
        # neuron_list = [t_spikes, d_spikes, b_spikes, o_spikes,moc_spikes]#,an_spikes]
        # neuron_title_list = ['d_stellate', 'bushy', 'octopus','moc','an']
        # neuron_list = [d_spikes, b_spikes, o_spikes,moc_spikes]#,an_spikes]
        # plt.figure("spikes ear{}".format(ear_index))
        # for i, neuron_times in enumerate(neuron_list):
        #     non_zero_neuron_times = neuron_times[ear_index]#[spikes for spikes in neuron_times[ear_index] if len(spikes)>0]#
        #     spike_raster_plot_8(non_zero_neuron_times, plt, duration/1000., len(non_zero_neuron_times) + 1, 0.001,
        #                         title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1)
        #                         )  # ,filepath=results_directory)
        #     # psth_plot_8(plt, numpy.arange(len(non_zero_neuron_times)), non_zero_neuron_times, bin_width=0.25 / 1000.,
        #     #             duration=duration / 1000.,title="PSTH_T ear{}".format(0))
        # middle_channel = int(len(moc_att[ear_index]) / 2.)
        # plt.figure("moc attenuation ear{} channel {}".format(ear_index, middle_channel))
        # moc_signal = moc_att[ear_index][middle_channel]
        # x = np.linspace(0, duration, len(moc_signal))
        # plt.plot(x, moc_signal)
        # plt.xlabel("time (ms)")
# noinspection PyUnboundLocalVariable
sim.end()
print "simulation of {}s complete in {}s".format(duration/1000.,local_time.time()-time_start)

# isi_cut_off = 20
# max_power = min([np.log10(Fs/2.),4.25])
# pole_freqs = np.logspace(np.log10(30),max_power,n_fibres/10)
# freq_index = int(np.where(pole_freqs==find_nearest(pole_freqs,freq))[0]*10)
# ten_pc = int(freq_index*0.1)
# left_t_spike_trains = [spike_train for spike_train in t_spikes[0][freq_index-ten_pc:freq_index+ten_pc] if len(spike_train)>0]
# t_isi = [isi(spike_train) for spike_train in left_t_spike_trains]
# t_isi_filtered = []
# for fibre in t_isi:
#     t_isi_filtered.append([i for i in fibre if i<isi_cut_off])
# # cvs = [cv(i) for i in t_isi]
# # cvs = [cv(i) for i in t_isi_filtered]
# # plt.figure("CV")
# # plt.hist(cvs)
# all_isi = []
# for fibre in t_isi:
#     for i in fibre:
#         if i.item()<isi_cut_off:
#             all_isi.append(i.item())
# plt.figure("ISI all")
# plt.hist(all_isi)

# plt.figure("ISI")
# chosen_isi = []
# n_choices = 20
# chosen_indices = np.random.choice(len(t_isi_filtered),n_choices,replace=False)
# for index in chosen_indices:
#     chosen_isi.append(t_isi_filtered[index])
# for i,neuron in enumerate(chosen_isi):
#     all_isi = [interval.item() for interval in neuron]
#     plt.subplot(n_choices/2,2,i+1)
#     plt.hist(all_isi)
#     plt.xlim((0,isi_cut_off))

# single_isi = isi(left_t_spike_trains[len(left_t_spike_trains)/2])
# single_isi = [i.item() for i in single_isi if i<isi_cut_off]
# plt.figure("ISI single")
# plt.hist(single_isi)

if record_en:
    # np.savez_compressed(input_directory+'/cn_' + test_file + '_{}an_fibres_{}ms_timestep_{}dB_{}s_moc_{}_lat_{}'.format
    #                      (number_of_inputs,timestep,dBSPL,int(duration/1000.),moc_feedback,lateral),an_spikes=an_spikes,
    #                      t_spikes=t_spikes,d_spikes=d_spikes,b_spikes=b_spikes,o_spikes=o_spikes,moc_spikes=moc_spikes,onset_times=onset_times,
    #                     moc_att=moc_att,Fs=Fs,stimulus=audio_data)
    np.savez_compressed(input_directory+'/cn_' + test_file + '_{}an_fibres_{}ms_timestep_{}dB_{}s_moc_{}_lat_{}'.format
                         (number_of_inputs,timestep,dBSPL,int(duration/1000.),moc_feedback,lateral),ear_data=ear_data,sg_data=sg_data,
                         t_data=t_data,d_data=d_data,b_data=b_data,o_data=o_data,moc_data=moc_data,onset_times=onset_times,
                         Fs=Fs,stimulus=audio_data)

    plt.show()