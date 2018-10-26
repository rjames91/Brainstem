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
inh_cond_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               #'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 10.0,#2.5,
               #'tau_syn_I': 1.0,#2.5,
               'e_rev_E': -50.0,
               'e_rev_I': 0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh':-45.0#-55.4#
               }

t_stellate_params_cond = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
                'tau_m': 3.8,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 0.94/0.37,#3.0,#2.5,#
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
               'tau_syn_E': 4.88/0.37,#2.5,#
               'e_rev_E': -15.,#-35.,#-25.,#-55.1,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,
               'v_rest': -63.0,
               'v_thresh': -39.5
               }



IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                   'v_init': -75,
                   'u_init': 10.,#0.,
                   }

dB = 50#40#20
duration = 1#75#
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
#cochlea_file = np.load(input_directory + '/spinnakear_13.5kHz_{}s_{}dB.npz'.format(duration,dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5kHz_38s_{}dB_1000fibres.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_1000fibres.npz'.format(dB))
cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_5000fibres.npz'.format(dB))
# cochlea_file = np.load(input_directory + '/spinnakear_13.5_1_kHz_75s_{}dB_10000fibres.npz'.format(dB))
an_spikes = cochlea_file['scaled_times']
onset_times = cochlea_file['onset_times']

a= np.logspace(1.477,4.3,100)

w2s_target = 0.4#0.5#1.#1.5#4.5#0.12#2.5#5.
input_spikes = an_spikes#[4000:6000]
number_of_inputs = len(input_spikes)#

spike_raster_plot_8(input_spikes,plt,duration,number_of_inputs+1,title="input activity")
plt.show()
#================================================================================================
# SpiNNaker setup
#================================================================================================
timestep =1.#0.1#
sim.setup(timestep=timestep)
# sim.set_number_of_neurons_per_core(sim.IF_cond_exp,8)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp,32)
# sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,32)

#================================================================================================
# Populations
#================================================================================================
#assume total number of CN neurons (CH + ON + PL) = 2 * n_inputs (taken from Moore 1990 Ferret total numbers)
# n_total = 5.5625 * number_of_inputs
n_total = 2. * number_of_inputs
#ratios taken from campagnola & manis 2014 mouse
n_ch = int(n_total * 2./3 * 24./89)
print "n_ch: {}".format(n_ch)
n_on = int(n_total * 1./3 * 24./89)
n_pl = int(n_total * 55./89) # TODO: may want to split this further into GBC and SBCs
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes),label="an_input_pop")
# inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
ch_pop = sim.Population(n_ch,sim.IF_cond_exp,ch_params_cond,label="ch_fixed_weight_scale_cond")
# on_ch_ratio = 0.1
on_pop_size = n_on#int(number_of_inputs*on_ch_ratio)
on_pop = sim.Population(on_pop_size,sim.IF_cond_exp,on_params_cond,label="on_fixed_weight_scale_cond")
# on_pop = sim.Population(on_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Onset")

# cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_curr_exp,inh_params,label="fixed_weight_scale")
# inh_pop =

ch_pop.record(["spikes","v"])
on_pop.record(["spikes","v"])
# ch_pop.record(["spikes"])
# on_pop.record(["spikes"])
#================================================================================================
# Projections
#================================================================================================
diagonal_width = 3.1423#2.24#2.#26.#
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(n_ch / number_of_inputs), 1)
sigma = dist * diagonal_width
conn_num = int(np.round(sigma / in2out_sparse))#Xie Manis 5-7
av_weight = w2s_target/conn_num
# an2ch_weight = RandomDistribution('normal',[av_weight,0.1])
# an2ch_weight = RandomDistribution('normal',[av_weight,0.5])
# an2ch_weight = RandomDistribution('uniform',[0,av_weight+(av_weight*1.5)])
# an2ch_weight = RandomDistribution('uniform',[0.9*(w2s_target/conn_num),1.1*(w2s_target/conn_num)])
an2ch_weight = RandomDistribution('normal_clipped',[av_weight/2.,av_weight*0.5,0.,1.1*av_weight])
# plt.hist(an2ch_weight.next(1000),bins=100)
# plt.show()
an2ch_list = normal_dist_connection_builder(number_of_inputs,n_ch,RandomDistribution,conn_num,dist,sigma)
pres = np.asarray([pre for (pre,post) in an2ch_list])
x= np.nonzero(pres==25)
posts = np.asarray([post for (pre,post) in an2ch_list])
y = np.nonzero(posts==25)

reduced_list = [(pre,post) for (pre,post) in an2ch_list if post == 500]
pres = [pre for (pre,post) in reduced_list]
an_ch_projection = sim.Projection(input_pop,ch_pop,sim.FromListConnector(an2ch_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))
# an_ch_projection = sim.Projection(input_pop,ch_pop,sim.FromListConnector(reduced_list),synapse_type=sim.StaticSynapse(weight=an2ch_weight))

#TODO: calculate formulas for these variables so we can scale
an_scale_factor = number_of_inputs/30000.
n_on_inputs = 10.#np.round(an_scale_factor*88.)#upper bound from Xie and Manis (2017)
an_on_p_connect = n_on_inputs/float(number_of_inputs)
# an_on_p_connect = 0.1#0.55
# an_on_weight = w2s_target/(n_on_inputs)
# an_on_weight = 1.5/(n_on_inputs)
av_weight = 0.05#0.05#0.015#0.005#0.002#0.00125#0.0055#0.005#
# av_weight = 4.#0.005#0.002#0.00125#0.0055#0.005#
# an_on_weight =RandomDistribution('uniform',[0.9*av_weight,1.1*av_weight])
an_on_weight =RandomDistribution('uniform',[0,av_weight])

# test = RandomDistribution('normal_clipped',[500,10*number_of_inputs,0,1000])
# plt.hist(test.next(1000),bins=1000)
# plt.show()
# an_on_weight =RandomDistribution('normal_clipped',[av_weight,av_weight/5.,0,2*av_weight])
# an_on_projection = sim.Projection(input_pop,on_pop,sim.FixedProbabilityConnector(p_connect=an_on_p_connect),sim.StaticSynapse(weight=an_on_weight))
an_on_list = normal_dist_connection_builder(number_of_inputs,n_on,RandomDistribution,conn_num=n_on_inputs,dist=1.,sigma=10*number_of_inputs
                                            ,conn_weight="hack",dist_weight=av_weight)
# pres = np.asarray([pre for (pre,post,weight,delay) in an_on_list])
# posts = np.asarray([post for (pre,post,weight,delay) in an_on_list])
# an_on_projection = sim.Projection(input_pop,on_pop,sim.FromListConnector(an_on_list),
#                                   synapse_type=sim.StaticSynapse())

#TODO: adjust these for scaling
inh_weight = RandomDistribution('uniform',[0.,0.25])
exc_weight = RandomDistribution('uniform',[0.,0.02])
# exc_weight = RandomDistribution('uniform',[0.,0.15])

# local_delays = RandomDistribution('uniform_int',[1,7])
local_delays = RandomDistribution('normal_clipped',[7,4,4,19])
# delays = local_delays.next(1000)
# plt.hist(delays,bins=100)
# plt.show()

#distance dependant interneuron connectivity
local_n_conn = 5.
local_dist = 1.
local_sigma = 1.

# on_on_list = normal_dist_connection_builder(n_on,n_on,RandomDistribution,conn_num=local_n_conn,dist=local_dist,sigma=local_sigma,p_connect=1./3,conn_weight=inh_weight,delay=local_delays,delay_scale=0.4)
# on_on_projection = sim.Projection(on_pop,on_pop,sim.FromListConnector(on_on_list),
#                                   synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
#
# on_ch_list = normal_dist_connection_builder(n_on,n_ch,RandomDistribution,conn_num=local_n_conn,dist=local_dist,sigma=local_sigma,p_connect=2./3,conn_weight=inh_weight,delay=local_delays,delay_scale=0.4)
# on_ch_projection = sim.Projection(on_pop,ch_pop,sim.FromListConnector(on_ch_list),
#                                   synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
#
# ch_ch_list = normal_dist_connection_builder(n_ch,n_ch,RandomDistribution,conn_num=local_n_conn,dist=local_dist,sigma=local_sigma,p_connect=2./3,conn_weight=exc_weight,delay=local_delays,delay_scale=0.4)
# ch_ch_projection = sim.Projection(ch_pop,ch_pop,sim.FromListConnector(ch_ch_list),
#                                   synapse_type=sim.StaticSynapse(weight=exc_weight))
#
# ch_on_list = normal_dist_connection_builder(n_ch,n_on,RandomDistribution,conn_num=local_n_conn,dist=local_dist,sigma=local_sigma,p_connect=1./3,conn_weight=exc_weight,delay=local_delays,delay_scale=0.4)
# ch_on_projection = sim.Projection(ch_pop,on_pop,sim.FromListConnector(ch_on_list),
#                                   synapse_type=sim.StaticSynapse(weight=exc_weight))


# on_on_projection = sim.Projection(on_pop,on_pop,sim.FixedProbabilityConnector(p_connect=p_local_connection),
#                                   synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
# on_ch_projection = sim.Projection(on_pop,ch_pop,sim.FixedProbabilityConnector(p_connect=p_local_connection),
#                                   synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')

# ch_on_projection = sim.Projection(ch_pop,on_pop,sim.FixedProbabilityConnector(p_connect=p_local_connection),
#                                  synapse_type=sim.StaticSynapse(weight=exc_weight))
# ch_ch_projection = sim.Projection(ch_pop,ch_pop,sim.FixedProbabilityConnector(p_connect=p_local_connection),
#                                  synapse_type=sim.StaticSynapse(weight=exc_weight))


duration_ms = duration*1000.#max(input_spikes[0])
max_period = 8000.#60000.#
num_recordings =int((duration_ms/max_period)+1)

for i in range(num_recordings):
    sim.run(duration_ms/num_recordings)

ch_data = ch_pop.get_data(["spikes","v"])
on_data = on_pop.get_data(["spikes","v"])
# ch_data = ch_pop.get_data(["spikes"])
# on_data = on_pop.get_data(["spikes"])
sim.end()

ch_spikes = ch_data.segments[0].spiketrains
ch_spikes_clean = []
for neuron in ch_spikes:
	ch_spikes_clean.append(np.asarray([time.item() for time in neuron]))

on_spikes = on_data.segments[0].spiketrains
on_spikes_clean = []
for neuron in on_spikes:
	on_spikes_clean.append(np.asarray([time.item() for time in neuron]))

psth_spikes = []#repeat_test_spikes_gen(ch_spikes,899,onset_times)
np.savez_compressed(input_directory+'/cn_13.5_1kHz_{}dB_{}ms_timestep_{}s'.format(dB,timestep,duration),an_spikes=an_spikes,ch_spikes=ch_spikes_clean,
                    on_spikes=on_spikes_clean,psth_spikes=psth_spikes,onset_times=onset_times)

print "results saved"
# ch_mem_v = ch_data.segments[0].filter(name='v')
# # cell_voltage_plot_8(ch_mem_v, plt, duration, [],id=599,scale_factor=0.001,title='ch pop')
# cell_voltage_plot_8(ch_mem_v, plt, duration_ms/timestep, [],id=None,scale_factor=timestep/1000.,title='ch pop')
# on_mem_v = on_data.segments[0].filter(name='v')
# # cell_voltage_plot_8(on_mem_v, plt, duration, [],id=59,scale_factor=0.001,title='on pop')
# cell_voltage_plot_8(on_mem_v, plt, duration_ms/timestep, [],id=None,scale_factor=timestep/1000.,title='on pop')

spike_raster_plot_8(ch_spikes,plt,duration,n_ch+1,title="ch pop activity")
# spike_raster_plot_8(on_spikes,plt,duration,on_pop_size+1,title="on pop activity")
spike_raster_plot_8(input_spikes,plt,duration,number_of_inputs+1,title="input activity")

#TODO: run this with 0.1ms bin width (will need to re-run sim at 0.1 time step + time scale factor=10)
#TODO: run repeat PSTH tests - need to copy file from laptop...

pres.sort()
print pres
# psth_plot_8(plt,pres,an_spikes,bin_width=0.001,duration=1.,title="PSTH_CH")
#psth_plot_8(plt,numpy.arange(len(psth_spikes[0])),psth_spikes[0],bin_width=0.0001,duration=0.1,title="PSTH_CH")

# psth_plot_8(plt,numpy.arange(550,650),an_spikes,bin_width=0.01,duration=duration/1000.,title="PSTH_AN")
# psth_plot_8(plt,numpy.arange(100,200),ch_spikes,bin_width=0.01,duration=duration,title="PSTH_CH")
# psth_plot_8(plt,numpy.arange(55,65),on_spikes,bin_width=0.01,duration=duration,title="PSTH_ON")

plt.show()