import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

#================================================================================================
# Simulation parameters
#================================================================================================
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

dB = 30
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
test_file = 'matches_6s_{}dB.npz'.format(dB)
# test_file = 'stereo_1k_1s_{}dB.npz'.format(dB)
cochlea_file = np.load(input_directory + '/spinnakear_' + test_file)
an_input = cochlea_file['scaled_times']

max_time = 0.
if an_input.shape[0]>1:
    n_ears =2
    for ear in an_input:
        for fibre in ear:
            for time in fibre:
                if time > max_time:
                    max_time = time

else:
    n_ears = 1
    for fibre in an_input:
        for time in fibre:
            if time > max_time:
                max_time = time

duration = max_time

input_pops = [[] for _ in range(n_ears)]
t_pops = [[] for _ in range(n_ears)]
d_pops = [[] for _ in range(n_ears)]

an_t_projs = [[] for _ in range(n_ears)]
an_d_projs = [[] for _ in range(n_ears)]
t_d_projs = [[] for _ in range(n_ears)]
t_t_projs = [[] for _ in range(n_ears)]
d_t_projs = [[] for _ in range(n_ears)]
d_d_projs = [[] for _ in range(n_ears)]
d_tc_projs = [[] for _ in range(n_ears)]
d_dc_projs = [[] for _ in range(n_ears)]

onset_times = cochlea_file['onset_times']

#================================================================================================
# SpiNNaker setup
#================================================================================================
timestep = 1.0#0.1#
sim.setup(timestep=timestep)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,64)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

for ear_index,an_spikes in enumerate(an_input):
    number_of_inputs = len(an_spikes)
    input_spikes = an_spikes

    n_total = 2. * number_of_inputs
    #ratios taken from campagnola & manis 2014 mouse
    n_t = int(n_total * 2./3 * 24./89)
    n_d = int(n_total * 1./3 * 24./89)

    #================================================================================================
    # AN --> CN Projections
    #================================================================================================
    w2s_t = 0.7
    n_an_t_connections = RandomDistribution('uniform',[4.,6.])
    av_an_t = w2s_t/5.
    # an_t_weight = RandomDistribution('uniform',[0,av_an_t])
    an_t_weight = RandomDistribution('normal_clipped',[av_an_t,0.1*av_an_t,0,av_an_t*2.])
    an_t_list = normal_dist_connection_builder(number_of_inputs,n_t,RandomDistribution,conn_num=n_an_t_connections,dist=1.,sigma=1.
                                                ,conn_weight=an_t_weight)
    # an_t_projection = sim.Projection(input_pop,t_pop,sim.FromListConnector(an_t_list),synapse_type=sim.StaticSynapse())
    input_pops[ear_index],t_pops[ear_index],an_t_projs[ear_index],m_pre = sub_pop_builder_inter(sim,n_t,sim.IF_cond_exp,t_stellate_params_cond,"SSA",input_spikes,
                                                                "an_input","t_stellate",an_t_list,post_record_list=["spikes","v"])

    # n_an_d_connections = RandomDistribution('uniform',[11.,88.])
    n_an_d_connections = RandomDistribution('normal_clipped',[60.,5.,11.,88.])
    w2s_d = 1.#0.5#
    av_an_d = w2s_d/60.#w2s_d/88.
    # an_d_weight = RandomDistribution('uniform',[0,av_an_d])
    an_d_weight = RandomDistribution('normal_clipped',[av_an_d,0.1*av_an_d,0,av_an_d*2.])
    an_d_list = normal_dist_connection_builder(number_of_inputs,n_d,RandomDistribution,conn_num=n_an_d_connections,dist=1.,
                                               # sigma=number_of_inputs/5.,conn_weight=an_d_weight)
                                               sigma=number_of_inputs/12.,conn_weight=an_d_weight)
    # an_d_projection = sim.Projection(input_pop,d_pop,sim.FromListConnector(an_d_list),synapse_type=sim.StaticSynapse())
    input_pops[ear_index],d_pops[ear_index],an_d_projs[ear_index],m_pre = sub_pop_builder_inter(sim,n_d,sim.IF_cond_exp,d_stellate_params_cond,"SSA",input_spikes,
                                                                "an_input","d_stellate",an_d_list,pre_pops=input_pops[ear_index],post_record_list=["spikes","v"])

#now all populations have been created we can create lateral projections
#================================================================================================
# Lateral CN Projections
#================================================================================================
for ear_index in range(n_ears):
    av_t_t = 0.1
    t_t_weight = RandomDistribution('normal_clipped',[av_t_t,0.1*av_t_t,0,av_t_t*2.])
    # plt.hist(t_t_weight.next(1000),bins=100)
    t_t_list = normal_dist_connection_builder(n_t,n_t,RandomDistribution,conn_num=10.,dist=1.,sigma=2.,conn_weight=t_t_weight)
    # t_t_projection = sim.Projection(t_pop,t_pop,sim.FromListConnector(t_t_list),synapse_type=sim.StaticSynapse())

    t_t_projs[ear_index]=sub_pop_projection_builder(t_pops[ear_index],t_pops[ear_index],t_t_list,sim)

    t_d_list = normal_dist_connection_builder(n_t,n_d,RandomDistribution,conn_num=10.,dist=1.,sigma=2.,conn_weight=t_t_weight)
    t_d_projs[ear_index]=sub_pop_projection_builder(t_pops[ear_index],d_pops[ear_index],t_d_list,sim)

    av_d_d = 0.1
    d_d_weight = RandomDistribution('normal_clipped',[av_d_d,0.1*av_d_d,0,av_d_d*2.])
    d_d_list = normal_dist_connection_builder(n_d,n_d,RandomDistribution,conn_num=10.,dist=1.,sigma=1.,conn_weight=d_d_weight)
    d_d_projs[ear_index]=sub_pop_projection_builder(d_pops[ear_index],d_pops[ear_index],d_d_list,sim,receptor_type='inhibitory')

    d_t_list = normal_dist_connection_builder(n_d,n_t,RandomDistribution,conn_num=10.,dist=1.,sigma=1.,conn_weight=d_d_weight)
    d_t_projs[ear_index]=sub_pop_projection_builder(d_pops[ear_index],t_pops[ear_index],d_t_list,sim,receptor_type='inhibitory')

    #TODO: verify contralateral d->tc and d->dc projection stats
    contra_ear_index = n_ears - 1 - ear_index
    d_tc_list = normal_dist_connection_builder(n_d,n_t,RandomDistribution,conn_num=10.,dist=1.,sigma=20.,conn_weight=d_d_weight)
    d_tc_projs[ear_index]=sub_pop_projection_builder(d_pops[ear_index],t_pops[contra_ear_index],d_tc_list,sim,receptor_type='inhibitory')

    d_dc_list = normal_dist_connection_builder(n_d,n_d,RandomDistribution,conn_num=10.,dist=1.,sigma=20.,conn_weight=d_d_weight)
    d_dc_projs[ear_index]=sub_pop_projection_builder(d_pops[ear_index],d_pops[contra_ear_index],d_dc_list,sim,receptor_type='inhibitory')

max_period = 6000.
num_recordings =int((duration/max_period)+1)

for i in range(num_recordings):
    sim.run(duration/num_recordings)

for ear_index in range(n_ears):
    t_spikes = get_sub_pop_spikes(t_pops[ear_index])
    d_spikes = get_sub_pop_spikes(d_pops[ear_index])

    spike_raster_plot_8(t_spikes,plt,duration/1000.,n_t+1,0.001,title="t stellate pop activity ear{}".format(ear_index))
    spike_raster_plot_8(d_spikes,plt,duration/1000.,n_d+1,0.001,title="d stellate pop activity ear{}".format(ear_index))
# spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")
sim.end()

if duration < 6000.:
    # mem_v = t_data.segments[0].filter(name='v')
    # cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,title='t stellate pop')
    # mem_v = d_data.segments[0].filter(name='v')
    # cell_voltage_plot_8(mem_v, plt, duration/timestep, [],scale_factor=timestep/1000.,title='d stellate pop')

    psth_plot_8(plt,numpy.arange(150,200),t_spikes,bin_width=timestep/1000.,duration=duration/1000.,title="PSTH_T")

np.savez_compressed(input_directory+'/chopper_' + test_file + '_{}an_fibres_{}ms_timestep_{}s'.format
                    (dB,number_of_inputs,timestep,int(duration/1000.)),an_spikes=an_spikes,
                    t_spikes=t_spikes,d_spikes=d_spikes,onset_times=onset_times)
plt.show()