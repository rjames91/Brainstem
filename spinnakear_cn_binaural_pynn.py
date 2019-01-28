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
t_stellate_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
                'tau_m': 3.8,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 0.94,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               'tau_syn_I': 4.0,#2.5,#
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
               'tau_syn_I': 4.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }

octopus_params_cond = {'cm': 1.,#57.,  # nF Only 200 cells in mouse CN
               'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 0.35,#2.5,#
               'e_rev_E': -55.,#-25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.6,#-70.0,
               'v_rest': -60.6,
               'v_thresh': -56.
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

bushy_params_cond = {#'cm': 5.,#57.,  # nF Only 200 cells in mouse CN
               #'tau_m': 0.5,#10.0,#2.,#3.,#
               'tau_syn_E': 2.,#2.5,#
               #'e_rev_E': -25.,#-10.,#-35.,#-55.1,#
               'v_reset': -60.,#-70.0,
               'v_rest': -60.,
               'v_thresh': -40.
               }
t_stellate_izk_class_2_params = {
               'a':0.2,
               'b':0.26,
               'c':-65,
               'd':10,#0,#vary between 0 and 10 for CRs 100-500Hz
               'u':-15,
               'tau_syn_E': 0.94,#3.0,#2.5,#
               #'e_rev_E': -54.,#-55.1,#
               'tau_syn_I': 4.0,#2.5,#
               'v': -63.0,
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
sub_pop = False
conn_pre_gen = True
lateral = True

Fs = 100000.#22050.#
dBSPL=60
wav_directory = '/home/rjames/SpiNNaker_devel/OME_SpiNN/'
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

tone_1,amp = generate_signal(freq=1000,dBSPL=dBSPL,duration=0.05,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_1_r,amp = generate_signal(freq=1000,dBSPL=0.,duration=0.05,
                       modulation_freq=0.,fs=Fs,ramp_duration=0.005,plt=None,silence=True,silence_duration=0.075)
tone_1_stereo = np.asarray([tone_1,tone_1_r])

timit_l,amp = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788.wav',plt=None,channel=0)
timit_r,amp = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788.wav',plt=None,channel=1,amp=amp)
timit = numpy.asarray([timit_l,timit_r])

sounds_dict = {
                "tone_1":tone_1,
                "tone_1_stereo":tone_1_stereo,
                "timit":timit
}
n_fibres = 1000
timestep = 1.0#0.1#
required_total_time = 1.

stimulus_list = ['timit']
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
d_pops = [[] for _ in range(n_ears)]
b_pops = [[] for _ in range(n_ears)]
o_pops = [[] for _ in range(n_ears)]
moc_pops = [[] for _ in range(n_ears)]

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

t_spikes = [[] for _ in range(n_ears)]
d_spikes = [[] for _ in range(n_ears)]
b_spikes = [[] for _ in range(n_ears)]
o_spikes = [[] for _ in range(n_ears)]
moc_spikes = [[] for _ in range(n_ears)]


if conn_pre_gen:
    connection_dicts_file = np.load(input_directory+'/cn_{}an_fibres_connectivity.npz'.format
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
    # n_total = 2. * number_of_inputs
    #ratios taken from campagnola & manis 2014 mouse
    n_t = int(n_total * 2./3 * 24./89)
    n_d = int(n_total * 1./3 * 24./89)
    n_b = int(n_total * 55./89)#number_of_inputs#
    n_o = int(n_total * 10./89.)
    n_moc = int(number_of_inputs * (360/30e3))

    #================================================================================================
    # Build Populations
    #================================================================================================
    pop_size = max([number_of_inputs,n_d,n_t,n_b,n_o,n_moc])
    if pop_size != number_of_inputs: # need to scale
        an_spatial=False
    else:
        an_spatial = True
    input_pops[ear_index]=sim.Population(number_of_inputs,SpiNNakEar(audio_input=audio_data[ear_index],fs=Fs,n_channels=number_of_inputs/10),label="AN_Pop_ear{}".format(ear_index))
    d_pops[ear_index]=sim.Population(pop_size,sim.extra_models.Izhikevich_cond,d_stellate_izk_class_1_params,label="d_stellate_fixed_weight_scale_cond{}".format(ear_index))
    # d_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,d_stellate_params_cond,label="d_stellate_fixed_weight_scale_cond".format(ear_index))
    # d_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,d_stellate_params_cond,label="d_stellate".format(ear_index))
    d_pops[ear_index].record(["spikes"])
    # t_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,t_stellate_params_cond,label="t_stellate_fixed_weight_scale_cond".format(ear_index))
    t_pops[ear_index]=sim.Population(pop_size,sim.extra_models.Izhikevich_cond,t_stellate_izk_class_2_params,label="t_stellate_fixed_weight_scale_cond{}".format(ear_index))
    # t_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,t_stellate_params_cond,label="t_stellate".format(ear_index))
    t_pops[ear_index].record(["spikes"])
    b_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,bushy_params_cond,label="bushy_fixed_weight_scale_cond{}".format(ear_index))
    # b_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,bushy_params_cond,label="bushy".format(ear_index))
    b_pops[ear_index].record(["spikes"])
    # o_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,octopus_params_cond,label="octopus_fixed_weight_scale_cond{}".format(ear_index))
    o_pops[ear_index]=sim.Population(pop_size,sim.extra_models.Izhikevich_cond,octopus_params_cond_izh,label="octopus_fixed_weight_scale_cond{}".format(ear_index))
    # o_pops[ear_index]=sim.Population(pop_size,sim.IF_cond_exp,octopus_params_cond,label="octopus".format(ear_index))
    o_pops[ear_index].record(["spikes"])
    # moc_pops[ear_index] = sim.Population(pop_size, sim.extra_models.Izhikevich_cond, IZH_EX_SUBCOR, label="moc_fixed_weight_scale_cond{}".format(ear_index))
    moc_pops[ear_index] = sim.Population(n_moc, sim.extra_models.Izhikevich_cond, IZH_EX_SUBCOR, label="moc_fixed_weight_scale_cond{}".format(ear_index))
    moc_pops[ear_index].set_constraint(MaxVertexAtomsConstraint(1))
    moc_pops[ear_index].record(["spikes"])

    #================================================================================================
    # AN --> CN Projections
    #================================================================================================
    # conn_lists = np.load('./conn_lists_{}.npz'.format(n_fibres))
    w2s_t = 0.3#0.7
    n_an_t_connections = RandomDistribution('uniform',[4.,6.])
    av_an_t = w2s_t/5.
    # an_t_weight = RandomDistribution('uniform',[0,av_an_t])
    an_t_weight = RandomDistribution('normal_clipped',[av_an_t,0.1*av_an_t,0,av_an_t*2.])
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

    # n_an_d_connections = RandomDistribution('uniform',[11.,88.])
    n_an_d_connections = RandomDistribution('normal_clipped',[60.,5.,11.,88.])
    w2s_d = 1.5#0.3#1.#0.5#
    av_an_d = w2s_d/60.#w2s_d/88.
    # an_d_weight = RandomDistribution('uniform',[0,av_an_d])
    an_d_weight = RandomDistribution('normal_clipped',[av_an_d,0.1*av_an_d,0,av_an_d*2.])
    if conn_pre_gen:
        an_d_list = connection_dicts[ear_index]['an_d_list']
    else:
        if an_spatial:
            an_d_list = spatial_normal_dist_connection_builder(pop_size,number_of_inputs,n_d,RandomDistribution,conn_num=n_an_d_connections,dist=1.,
                                               sigma=number_of_inputs/15.,conn_weight=an_d_weight)
        else:
            an_d_list = normal_dist_connection_builder(number_of_inputs,n_d,RandomDistribution,conn_num=n_an_d_connections,dist=1.,
                                               sigma=number_of_inputs/15.,conn_weight=an_d_weight)
            #scale up post ids
            scale_factor = float(pop_size-1)/(n_d-1)
            an_d_list = [(an,int(d*scale_factor),w,delay) for (an,d,w,delay) in an_d_list]

    an_d_projs[ear_index] = sim.Projection(input_pops[ear_index],d_pops[ear_index],sim.FromListConnector(an_d_list),synapse_type=sim.StaticSynapse())

    n_an_b_connections = RandomDistribution('uniform',[2.,5.])
    w2s_b = 0.3#
    av_an_b = w2s_b/2.
    an_b_weight = RandomDistribution('normal_clipped',[av_an_b,0.1*av_an_b,0,av_an_b*2.])
    if conn_pre_gen:
        an_b_list = connection_dicts[ear_index]['an_b_list']
    else:
        if an_spatial:
            an_b_list = spatial_normal_dist_connection_builder(pop_size,number_of_inputs,n_b,RandomDistribution,conn_num=n_an_b_connections,dist=1.,
                                               sigma=1.,conn_weight=an_b_weight)
        else:
            an_b_list = normal_dist_connection_builder(number_of_inputs,n_b,RandomDistribution,conn_num=n_an_b_connections,dist=1.,
                                               sigma=1.,conn_weight=an_b_weight)
            #scale up post ids
            scale_factor = float(pop_size-1)/(n_b-1)
            an_b_list = [(an,int(b*scale_factor),w,delay) for (an,b,w,delay) in an_b_list]

    an_b_projs[ear_index] = sim.Projection(input_pops[ear_index], b_pops[ear_index], sim.FromListConnector(an_b_list),
                                       synapse_type=sim.StaticSynapse())
    w2s_o = 3.#2.#7.
    # w2s_o = 5.
    n_an_o_connections = RandomDistribution('uniform',[30.,120.])
    # n_an_o_connections = RandomDistribution('normal_clipped',[50.,5.,30.,120.])
    av_an_o = w2s_o/50.
    an_o_weight = RandomDistribution('normal_clipped', [av_an_o, 0.1 * av_an_o, 0, av_an_o * 2.])
    if conn_pre_gen:
        an_o_list = connection_dicts[ear_index]['an_o_list']
    else:
        if an_spatial:
            an_o_list = spatial_normal_dist_connection_builder(pop_size, number_of_inputs, n_o, RandomDistribution,
                                                       conn_num=n_an_o_connections, dist=1.,
                                                       sigma=pop_size/20., conn_weight=an_o_weight)
        else:
            an_o_list = normal_dist_connection_builder(number_of_inputs, n_o, RandomDistribution,
                                                       conn_num=n_an_o_connections, dist=1.,
                                                       sigma=pop_size/20., conn_weight=an_o_weight)
            #scale up post ids
            scale_factor = float(pop_size-1)/(n_o-1)
            an_o_list = [(an,int(o*scale_factor),w,delay) for (an,o,w,delay) in an_o_list]

    an_o_projs[ear_index] = sim.Projection(input_pops[ear_index], o_pops[ear_index], sim.FromListConnector(an_o_list),
                                           synapse_type=sim.StaticSynapse())

    if conn_pre_gen is False:
        connection_dicts[ear_index]['an_t_list']=an_t_list
        connection_dicts[ear_index]['an_d_list']=an_d_list
        connection_dicts[ear_index]['an_o_list']=an_o_list
        connection_dicts[ear_index]['an_b_list']=an_b_list

#now all populations have been created we can create lateral projections
#================================================================================================
# Lateral CN Projections
#================================================================================================
n_lateral_connections = 100.
lateral_connection_strength = 0.1#0.3#0.6
d_t_ratio = float(n_d)/n_t
t_b_ratio = float(n_t)/n_b
d_b_ratio = float(n_d)/n_b
inh_ratio = 0.1#1.

t_lat_sigma = n_total * 0.01
d_lat_sigma = n_total * 0.1
b_lat_sigma = n_total * 0.01
# t_lat_sigma = 2.
# d_lat_sigma = 1.
# b_lat_sigma = 1.


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
        if conn_pre_gen:
            t_d_list = connection_dicts[ear_index]['t_d_list']
        else:
            t_d_list = spatial_normal_dist_connection_builder(pop_size,n_t,n_d,RandomDistribution,conn_num=d_t_ratio*n_lateral_connections,dist=1.,sigma=d_lat_sigma,conn_weight=t_t_weight)
        t_d_projs[ear_index] = sim.Projection(t_pops[ear_index],d_pops[ear_index],sim.FromListConnector(t_d_list),synapse_type=sim.StaticSynapse())
        if conn_pre_gen:
            t_b_list = connection_dicts[ear_index]['t_b_list']
        else:
            t_b_list = spatial_normal_dist_connection_builder(pop_size,n_t,n_b,RandomDistribution,conn_num=t_b_ratio*n_lateral_connections,dist=1.,sigma=b_lat_sigma,conn_weight=t_t_weight)
        t_b_projs[ear_index] = sim.Projection(t_pops[ear_index],b_pops[ear_index],sim.FromListConnector(t_b_list),synapse_type=sim.StaticSynapse())

        av_d_d = inh_ratio*(lateral_connection_strength/n_lateral_connections)#0.5#0.1#
        d_d_weight = RandomDistribution('normal_clipped',[av_d_d,0.1*av_d_d,0,av_d_d*2.])
        if conn_pre_gen:
            d_d_list = connection_dicts[ear_index]['d_d_list']
        else:
            d_d_list = spatial_normal_dist_connection_builder(pop_size,n_d,n_d,RandomDistribution,conn_num=d_t_ratio*n_lateral_connections,dist=1.,sigma=d_lat_sigma,conn_weight=d_d_weight)
        # d_d_projs[ear_index] = sim.Projection(d_pops[ear_index],d_pops[ear_index],sim.FromListConnector(d_d_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
        if conn_pre_gen:
            d_t_list = connection_dicts[ear_index]['d_t_list']
        else:
            d_t_list = spatial_normal_dist_connection_builder(pop_size,n_d,n_t,RandomDistribution,conn_num=d_t_ratio*n_lateral_connections,dist=1.,sigma=t_lat_sigma,conn_weight=d_d_weight)
        d_t_projs[ear_index] = sim.Projection(d_pops[ear_index],t_pops[ear_index],sim.FromListConnector(d_t_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
        if conn_pre_gen:
            d_b_list = connection_dicts[ear_index]['d_b_list']
        else:
            d_b_list = spatial_normal_dist_connection_builder(pop_size,n_d,n_b,RandomDistribution,conn_num=d_b_ratio*n_lateral_connections,dist=1.,sigma=b_lat_sigma,conn_weight=d_d_weight)
        d_b_projs[ear_index] = sim.Projection(d_pops[ear_index],b_pops[ear_index],sim.FromListConnector(d_b_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')

        #TODO: verify contralateral d->tc and d->dc projection stats
        if n_ears>1:
            contra_ear_index = n_ears - 1 - ear_index
            if conn_pre_gen:
                d_tc_list = connection_dicts[ear_index]['d_tc_list']
            else:
                d_tc_list = spatial_normal_dist_connection_builder(pop_size,n_d,n_t,RandomDistribution,conn_num=d_t_ratio*n_lateral_connections,dist=1.,sigma=t_lat_sigma,conn_weight=d_d_weight)
                connection_dicts[ear_index]['d_tc_list'] = d_tc_list

            d_tc_projs[ear_index]=sim.Projection(d_pops[ear_index],t_pops[contra_ear_index],sim.FromListConnector(d_tc_list),synapse_type=sim.StaticSynapse(),receptor_type='inhibitory')
            # ================================================================================================
            # CN --> VNTB Projections
            # ================================================================================================
            w2s_moc = 0.8
            n_t_moc_connections = RandomDistribution('uniform', [5, 10])
            av_t_moc = w2s_moc / 7.
            t_moc_weight = RandomDistribution('normal_clipped', [av_t_moc, 0.1 * av_t_moc, 0, av_t_moc * 2.])
            if conn_pre_gen:
                t_mocc_list = connection_dicts[ear_index]['t_mocc_list']
            else:
                t_mocc_list = normal_dist_connection_builder(n_t, n_moc, RandomDistribution,
                                                                     conn_num=n_t_moc_connections, dist=1.,
                                                                     sigma= n_t/ 100., conn_weight=t_moc_weight)
                # scale up pre ids
                scale_factor = float(pop_size - 1) / (n_t - 1)
                t_mocc_list = [(int(t * scale_factor),moc, w, delay) for (t, moc, w, delay) in t_mocc_list]
                connection_dicts[ear_index]['t_mocc_list'] = t_mocc_list

            t_mocc_projs[ear_index] = sim.Projection(t_pops[ear_index], moc_pops[contra_ear_index],
                                                     sim.FromListConnector(t_mocc_list),
                                                     synapse_type=sim.StaticSynapse())
            #ipsilateral moc
            n_moc_an_connections = RandomDistribution('uniform', [5, 10])
            moc_an_weight = 1.
            if conn_pre_gen:
                moc_an_list = connection_dicts[ear_index]['moc_an_list']
            else:
                moc_an_list = normal_dist_connection_builder(n_moc, number_of_inputs/10, RandomDistribution,
                                                                     conn_num=n_moc_an_connections, dist=1.,
                                                                     sigma=1., conn_weight=moc_an_weight)

                connection_dicts[ear_index]['moc_an_list'] = moc_an_list

            moc_an_projs[ear_index] = sim.Projection(moc_pops[ear_index], input_pops[ear_index],
                                                     sim.FromListConnector(moc_an_list),
                                                     synapse_type=sim.StaticSynapse())
            #contralateral moc
            if conn_pre_gen:
                moc_anc_list = connection_dicts[ear_index]['moc_anc_list']
            else:
                moc_anc_list = normal_dist_connection_builder(n_moc, number_of_inputs/10, RandomDistribution,
                                                                     conn_num=n_moc_an_connections, dist=1.,
                                                                     sigma=number_of_inputs / 100., conn_weight=moc_an_weight)
                connection_dicts[ear_index]['moc_anc_list'] = moc_an_list
            moc_anc_projs[ear_index] = sim.Projection(moc_pops[ear_index], input_pops[contra_ear_index],
                                                     sim.FromListConnector(moc_anc_list),
                                                     synapse_type=sim.StaticSynapse())

        if conn_pre_gen is False:
            connection_dicts[ear_index]['t_t_list'] = t_t_list
            connection_dicts[ear_index]['t_d_list'] = t_d_list
            connection_dicts[ear_index]['t_b_list'] = t_b_list
            connection_dicts[ear_index]['d_d_list'] = d_d_list
            connection_dicts[ear_index]['d_t_list'] = d_t_list
            connection_dicts[ear_index]['d_b_list'] = d_b_list

if conn_pre_gen is False:
    np.savez_compressed(input_directory+'/cn_{}an_fibres_connectivity'.format
                    (number_of_inputs),connection_dicts=connection_dicts)

max_period = 6000.
num_recordings =1#int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

for ear_index in range(n_ears):
    t_data = t_pops[ear_index].get_data(["spikes"])
    t_spikes[ear_index] = t_data.segments[0].spiketrains
    # mem_v = t_data.segments[0].filter(name='v')
    # cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,
    #                     title='t stellate pop ear{}'.format(ear_index),id=range(n_t))

    d_data = d_pops[ear_index].get_data(["spikes"])
    d_spikes[ear_index] = d_data.segments[0].spiketrains
    # mem_v = d_data.segments[0].filter(name='v')
    # cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,
    #                     title='d stellate pop ear{}'.format(ear_index),id=range(n_d))
    b_data = b_pops[ear_index].get_data(["spikes"])
    b_spikes[ear_index] = b_data.segments[0].spiketrains
    o_data = o_pops[ear_index].get_data(["spikes"])
    o_spikes[ear_index] = o_data.segments[0].spiketrains
    moc_data = moc_pops[ear_index].get_data(["spikes"])
    moc_spikes[ear_index] = moc_data.segments[0].spiketrains
    # if duration < 6000.:
    #     # psth_plot_8(plt, numpy.arange(175, 225), t_spikes, bin_width=timestep / 1000., duration=duration / 1000.,
    #     #             title="PSTH_T ear{}".format(ear_index))
    #
    #     # for pop in t_pops[ear_index]:
    #     #     data = pop.get_data(["v"])
    #     #     mem_v = data.segments[0].filter(name='v')
    #     #     cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,
    #     #                         title='t stellate pop ear{}'.format(ear_index),id=range(pop.size))

    plot_t_spikes = t_spikes#[spikes for spikes in t_spikes if len(spikes)>0]
    plot_d_spikes = d_spikes#[spikes for spikes in d_spikes if len(spikes)>0]
    plot_b_spikes = b_spikes#[spikes for spikes in b_spikes if len(spikes)>0]
    plot_o_spikes = o_spikes#[spikes for spikes in b_spikes if len(spikes)>0]
    plt_moc_spikes = moc_spikes
    # plot_an_spikes = scaled_input_spikes

    neuron_title_list = ['t_stellate', 'd_stellate', 'bushy', 'octopus','moc']
    neuron_list = [t_spikes, d_spikes, b_spikes, o_spikes,moc_spikes]
    plt.figure("spikes ear{}".format(ear_index))
    for i, neuron_times in enumerate(neuron_list):
        non_zero_neuron_times = neuron_times[ear_index]#[spikes for spikes in neuron_times[ear_index] if len(spikes)>0]#
        spike_raster_plot_8(non_zero_neuron_times, plt, duration/1000., len(non_zero_neuron_times) + 1, 0.001,
                            title=neuron_title_list[i], markersize=1, subplots=(len(neuron_list), 1, i + 1)
                            )  # ,filepath=results_directory)

    # spike_trains = [spikes for spikes in t_spikes[ear_index] if len(spikes>0)]
    # half_point = len(spike_trains)/2
    # t_isi = [isi(spike_train) for spike_train in spike_trains[half_point]]
    # isi_test = [isi.item() for isi in t_isi]
    # plt.figure("ISI ear {}".format(ear_index))
    # plt.hist(isi_test,bins=100)
    # spike_raster_plot_8(plot_t_spikes,plt,duration/1000.,pop_size+1,0.001,title="t stellate pop activity ear{}".format(ear_index))
    # spike_raster_plot_8(plot_d_spikes,plt,duration/1000.,pop_size+1,0.001,title="d stellate pop activity ear{}".format(ear_index))
    # spike_raster_plot_8(plot_b_spikes,plt,duration/1000.,pop_size+1,0.001,title="bushy pop activity ear{}".format(ear_index))
    # spike_raster_plot_8(plot_o_spikes,plt,duration/1000.,pop_size+1,0.001,title="octopus pop activity ear{}".format(ear_index))
    # spike_raster_plot_8(plot_an_spikes,plt,duration/1000.,pop_size+1,0.001,title="AN activity ear{}".format(ear_index))

sim.end()
print "simulation of {}s complete in {}s".format(duration/1000.,local_time.time()-time_start)

np.savez_compressed(input_directory+'/cn_' + test_file + '_{}an_fibres_{}ms_timestep_{}dB_{}s'.format
                     (number_of_inputs,timestep,dBSPL,int(duration/1000.)),an_spikes=[],
                     t_spikes=t_spikes,d_spikes=d_spikes,b_spikes=b_spikes,o_spikes=o_spikes,onset_times=onset_times)

plt.show()