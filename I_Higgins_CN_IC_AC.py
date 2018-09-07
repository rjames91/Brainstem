#import spynnaker7.pyNN as sim
import spynnaker8 as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import random

psth = True#False#
en_stdp = True #False#
# Setup pyNN simulation
timestep =1.#0.5#
sim.setup(timestep=timestep,max_delay=51.,min_delay=timestep)
# sim.set_number_of_neurons_per_core(sim.Izhikevich, 128)
sim.set_number_of_neurons_per_core(sim.Izhikevich, 64)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 16)
#sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#open AN spike source
#[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_no5.npy")
#[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_10sp_2num_5rep.npy")
# [spike_trains,an_scale_factor]=numpy.load("../OME_SpiNN/spike_trains_1sp_1num_150rep.npy")
# num_patterns =2
# stim_times = numpy.load('../../OME_SpiNN/spike_times_1sp_{}num_300rep.npy'.format(num_patterns))
#stim_times = numpy.load('../OME_SpiNN/spike_trains_yes.npy')
results_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/AN_spikes"
#stim_times=numpy.load(results_directory+'/spike_trains_asc_train_15s.npy')

ear_file = numpy.load(results_directory + '/spinnakear_asc_des_60s_60dB.npz')
stim_times = ear_file['scaled_times']
onset_times = ear_file['onset_times']
dBSPL = ear_file['dBSPL']


duration=0
for train in stim_times:
    if train.size>0 and train.max()>duration:
        duration = train.max()
sim_duration=duration.item()

AN_pop_size = len(stim_times)#numpy.max(spike_ids)#1000#
AC_pop_size = 128 #* AN_pop_size
# spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
# #an_scale_factor = 1./Fs#duration/numpy.max(spike_times)
# scaled_times = [spike_time * an_scale_factor for spike_time in spike_times]
# sim_duration =numpy.max(scaled_times)*1000#12*1000.#
# num_repeats = int(numpy.ceil(sim_duration/2000))#int(numpy.ceil(sim_duration/4000))#0#
num_repeats = int(numpy.ceil(sim_duration/500))
#
# single_duration= numpy.max(scaled_times)*1000
# spikes_train_an_ms = [(neuron_id,int(1000*an_scale_factor*spike_time)) for (neuron_id,spike_time) in spike_trains]

#[spikes_train_an_ms,duration] = numpy.load('./spike_times_an_alt.npy')
#sim_duration = duration * 1000.#600.#
#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

# for i in range(AN_pop_size):
#        id_times = [1000*timestep*an_scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i and spike_time<=300000]
#        spike_times_spinn.append(id_times)
# numpy.save("./spike_times_spinn_1sp_1num_150rep",spike_times_spinn)

# spike_times_spinn =numpy.load("./spike_times_spinn_1sp_1num_150rep.npy")# numpy.load("./spike_times_spinn_10sp_2num_5rep.npy")#("./spike_times_spinn_an_alt.npy")#("./spike_times_spinn_yes4.npy")
# sim_duration =numpy.max(spike_times_spinn)
# spike_times_spinn = numpy.load("./ic_spikes.npy")
AN_pop = sim.Population(AN_pop_size, sim.SpikeSourceArray, {'spike_times': stim_times}, label='AN pop')
# AN_pop = sim.Population(AN_pop_size, sim.SpikeSourceArray, {'spike_times': spike_times_spinn}, label='AN pop')
# spike_raster_plot_8(spike_times_spinn,plt=plt,duration=sim_duration/1000.,ylim=1001,scale_factor=0.001,title='AN')
# plt.show()
#params taken from I Higgins thesis pg. 121
IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                   'v_init': -75,
                   'u_init': 10.,#0.,
                   }

IZH_EX_COR = {'a': 0.01,
                   'b': 0.2,
                   'c': -65,
                   'd': 8,
                   }

IZH_INH = {'a': 0.02,
                   'b': 0.25,
                   'c': -55,
                   'd': 0.05,
                   }
LIF_cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 2.,#3.,#10.0,
               'tau_refrac': 1.0,#2.0,#
               'tau_syn_E': 1.0,#2.5,
               'tau_syn_I': 1.0,#2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

#create populations==================================================================
CH_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Chopper")
CH_pop_inh = sim.Population(AN_pop_size,sim.Izhikevich,IZH_INH,label="CN_Chopper_inh")
PL_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Primary like")
ON_pop = sim.Population(int(AN_pop_size/10.),sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Onset")
ON_pop_inh = sim.Population(int(AN_pop_size/10.),sim.Izhikevich,IZH_INH,label="CN_Onset_inh")
IC_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="IC")

VNTB_pop_size = 200#170
VNTB_pop = sim.Population(VNTB_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="VNTB")

# AC_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="AC")
# AC_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="AC_inh")
# Belt_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="Belt")
# Belt_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="Belt_inh")
# Belt2_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="Belt2")
# Belt2_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="Belt2_inh")

# AC_pop = sim.Population(AC_pop_size ,sim.IF_curr_exp,LIF_cell_params,label="target")
# AC_pop_inh = sim.Population(AC_pop_size ,sim.IF_curr_exp,LIF_cell_params,label="AC_inh")

#AN --> CH connectivity==================================================================
an2ch_weight = RandomDistribution('uniform',(0.5,0.8))#[0.7,0.8])#[0.,2.])#[1.,2.])#10.#25.
#an2ch_fan = 5
# diagonal_width = 26.
diagonal_width = AN_pop_size/38.46

diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AN_pop_size / AN_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
an2ch_list = normal_dist_connection_builder(AN_pop_size,AN_pop_size,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma,an2ch_weight)
#an2ch_proj = sim.Projection(AN_pop,CH_pop,sim.FromListConnector(an2ch_list))

#CH --> CH_inh connectivity==================================================================
ch2chinh_proj = sim.Projection(CH_pop,CH_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=0.5))
#CH_inh --> CH connectivity==================================================================
chinh2ch_proj = sim.Projection(CH_pop_inh,CH_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1),
                               receptor_type='inhibitory')

#AN --> ON connectivity==================================================================
an2on_prob=400./AN_pop_size
#an2on_prob=0.4#0.15#0.6#0.54#
an2on_weight =RandomDistribution('uniform',(0.05,0.15))
#an2on_proj = sim.Projection(AN_pop,ON_pop,sim.FixedProbabilityConnector(p_connect=an2on_prob,allow_self_connections=False),
#                            synapse_type=sim.StaticSynapse(weight=an2on_weight,delay=1))
#ON --> ON_inh connectivity==================================================================
on2oninh_proj = sim.Projection(ON_pop,ON_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=15.,delay=1.))
oninh2on_proj = sim.Projection(ON_pop_inh,ON_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.3,delay=1.),
                               receptor_type='inhibitory')

#AN --> PL connectivity==================================================================
an2pl_weight =RandomDistribution('uniform',(25.,40.))#[25.,35.])#
an2pl_proj = sim.Projection(AN_pop,PL_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=an2pl_weight,delay=1.0))#(weights=40.,delays=1.0))#

#CH --> IC connectivity==================================================================
ch2ic_weight = RandomDistribution('uniform',(10.,12.))#[4.,5.])
# diagonal_width = 2.
diagonal_width = AN_pop_size/500.
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AN_pop_size / AN_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
ch2ic_list = normal_dist_connection_builder(AN_pop_size,AN_pop_size,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma,ch2ic_weight)
ch2ic_proj = sim.Projection(CH_pop,IC_pop,sim.FromListConnector(ch2ic_list))

#CH --> VNTB connectivity==================================================================
ch2vntb_weight = RandomDistribution('uniform',(10.,15.))#20.#
# diagonal_width = 2.24#AN_pop_size/VNTB_pop_size#
diagonal_width = AN_pop_size/446.4
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AN_pop_size/VNTB_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
ch2vntb_list = normal_dist_connection_builder(AN_pop_size,VNTB_pop_size,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma,ch2vntb_weight)
ch2vntb_proj = sim.Projection(CH_pop,VNTB_pop,sim.FromListConnector(ch2vntb_list))

#ON --> IC connectivity==================================================================
on2ic_weight = RandomDistribution('uniform',(0.01,0.05))#[0,0.1])
on2ic_proj = sim.Projection(ON_pop,IC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=on2ic_weight))

#PL --> IC connectivity==================================================================
# pl2ic_weight = RandomDistribution('uniform',(10.,12.))
pl2ic_weight = RandomDistribution('uniform',(10.,15.))
pl2ic_proj = sim.Projection(PL_pop,IC_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=pl2ic_weight,delay=1.0))

#IC-->AC connectivity==================================================================
tau_factor = 1./timestep
tau_plus=16
tau_minus=30.
a_plus =0.01#0.001#
a_minus =0.01#0.001#

w2s_target = 5.
w_max = 1.#w2s/2.#2*w2s/num_pattern_neurons#w2s/10#w2s/stim_size#1.#1.#
w_min = 0.#w2s_target/num_pattern_neurons#
#start_weight = w_max/2.#
num_pattern_neurons = 10
av_weight = w2s_target/num_pattern_neurons#w_max/2.# w_max/3.#
ten_perc = av_weight/10.
start_weight = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))

stdp_weights = RandomDistribution('uniform',(av_weight-ten_perc,av_weight+ten_perc))
stdp_delays = RandomDistribution('uniform',(1,51.))

time_dep =sim.extra_models.SpikeNearestPairRule(tau_plus=tau_plus, tau_minus=tau_minus,
#time_dep =sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                          A_plus=a_plus, A_minus=a_minus)
weight_dep = sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max)
stdp = sim.STDPMechanism(time_dep,weight_dep,weight=stdp_weights,delay=stdp_delays)


print("************ ------------ using stdp ------------ ***********")
print("tau_plus = %f\ttau_minus = %f"% (tau_plus, tau_minus))
print("w_min = %f\tw_max = %f\ta_plus = %f\ta_minus = %f" %
      (w_min, w_max, a_plus, a_minus))

#aiming for around 50 post connections (one for each delay)
#num_post_conn = 50.#10.#1000.
#p_connect = num_post_conn/AC_pop_size #1.# 0.005 # 0.05
#num_incoming_connections = (num_post_conn * AN_pop_size)/AC_pop_size#AN_pop_size#
p_connect = 0.35
fixed_type = sim.StaticSynapse(weight=w2s_target/4.,delay=stdp_delays)

# ic2ac_proj = sim.Projection(
#     IC_pop, AC_pop, sim.FixedProbabilityConnector(p_connect=p_connect),
#         synapse_type=stdp,receptor_type="excitatory", source=None, space=None)
#         #synapse_type=fixed_type,receptor_type="excitatory", source=None, space=None)
# weights = ic2ac_proj.get("weight", "list", with_address=True)

#ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.AllToAllConnector(),synapse_type=stdp,receptor_type='excitatory')
# ic2ac_proj = sim.Projection(AN_pop,AC_pop,sim.AllToAllConnector(),synapse_type=stdp,receptor_type='excitatory')

#ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.OneToOneConnector(weights=30.,delays=ic2ac_delays),
#                            target='excitatory',synapse_dynamics=syn_dyn)
diagonal_width = 25.
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AC_pop_size / AN_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
# ic2ac_list = normal_dist_connection_builder(AN_pop_size,AC_pop_size,RandomDistribution,NumpyRNG(),
#                                             conn_num,dist,sigma,ic2ac_weights,delay=ic2ac_delays)
#ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.FromListConnector(ic2ac_list),target='excitatory',synapse_dynamics=syn_dyn)

#AC --> AC_inh connectivity
# ac2acinh_proj = sim.Projection(AC_pop,AC_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
# #acinh2ac_proj = sim.Projection(AC_pop_inh,AC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.005,delay=1.),
# acinh2ac_proj = sim.Projection(AC_pop_inh,AC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1,delay=1.),
#                                receptor_type='inhibitory')
#acinh2ac_proj = sim.Projection(AC_pop_inh,AC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.001,delay=1.),
#                               receptor_type='inhibitory')
#AC-->Belt connectivity
#num_incoming_connections = (num_post_conn * AC_pop_size)/AC_pop_size#AC_pop_size#
# num_post_conn = 10.
# p_connect = num_post_conn/AC_pop_size #1.# 0.005 # 0.05
# av_weight = w_max/2.#w2s/170.#w2s/4.#w_max/2#40./num_incoming_connections
# ten_perc = av_weight/10.
# ac2belt_weights = RandomDistribution('uniform',parameters=[av_weight-ten_perc,av_weight+ten_perc])#[0,max_weight])#w_max])
# ac2belt_delays = RandomDistribution('uniform',parameters=[1,51])
#ac2belt_proj = sim.Projection(AC_pop,Belt_pop,sim.FixedProbabilityConnector(p_connect=p_connect,weights=ac2belt_weights,delays=ac2belt_delays),
#                            target='excitatory',synapse_dynamics=syn_dyn)
# ac2belt_proj = sim.Projection(AC_pop,Belt_pop,sim.AllToAllConnector(weights=ac2belt_weights,delays=ac2belt_delays),
#                             target='excitatory',synapse_dynamics=syn_dyn)
# #Belt --> Belt_inh connectivity
# belt2beltinh_proj = sim.Projection(Belt_pop,Belt_pop_inh,sim.OneToOneConnector(weights=1.0,delays=1.))
# beltinh2belt_proj = sim.Projection(Belt_pop_inh,Belt_pop,sim.AllToAllConnector(weights=0.005,delays=1.),target='inhibitory')

#Belt-->Belt2 connectivity
# num_post_conn = 10.
# p_connect = num_post_conn/AC_pop_size #1.# 0.005 # 0.05
# max_weight = w2s/4.#w_max/2#40./num_incoming_connections
# ten_perc = max_weight/10.
# belt2belt2_weights = RandomDistribution('uniform',parameters=[max_weight-ten_perc,max_weight+ten_perc])#[0,max_weight])#w_max])
# belt2belt2_delays = RandomDistribution('uniform',parameters=[1,51])
# #belt2belt2_proj = sim.Projection(Belt_pop,Belt2_pop,sim.FixedProbabilityConnector(p_connect=p_connect,weights=ac2belt_weights,delays=ac2belt_delays),
# #                            target='excitatory',synapse_dynamics=syn_dyn)
# #Belt --> Belt_inh connectivity
# belt2belt2inh_proj = sim.Projection(Belt2_pop,Belt2_pop_inh,sim.OneToOneConnector(weights=1.0,delays=1.))
# beltinh2belt2_proj = sim.Projection(Belt2_pop_inh,Belt2_pop,sim.AllToAllConnector(weights=0.005,delays=1.),target='inhibitory')

#setup recordings
CH_pop.record('spikes')
# #CH_pop.record_v()
# #CH_pop.record_gsyn()
# CH_pop_inh.record('spikes')
ON_pop.record('spikes')
# #ON_pop.record_v()
# #ON_pop.record_gsyn()
#
# ON_pop_inh.record('spikes')
# #ON_pop_inh.record_v()
# #ON_pop_inh.record_gsyn()
#
PL_pop.record('spikes')
# #PL_pop.record_v()
VNTB_pop.record('spikes')
IC_pop.record('spikes')

# AC_pop.record('spikes')
# AC_pop_inh.record('spikes')

# Belt_pop.record('spikes')
# Belt_pop_inh.record('spikes')
#
# Belt2_pop.record('spikes')

# Run simulation
if num_repeats>1:
    varying_weights = []
    run_one = True

    for i in range(num_repeats):

        sim.run(sim_duration/num_repeats)

        CH_data = CH_pop.get_data("spikes")
        # # CH_v = CH_pop.get_v()
        # # CH_g = CH_pop.get_gsyn()
        # CHinh_data = CH_pop_inh.get_data("spikes")
        #
        ON_data = ON_pop.get_data("spikes")
        # # ON_v = ON_pop.get_v()
        # # ON_g = ON_pop.get_gsyn()
        #
        # ONinh_data = ON_pop_inh.get_data("spikes")
        # # ONinh_v = ON_pop_inh.get_v()
        # # ONinh_g = ON_pop_inh.get_gsyn()
        #
        PL_data = PL_pop.get_data("spikes")
        # # PL_v = PL_pop.get_v()
        #
        VNTB_data = VNTB_pop.get_data("spikes")
        IC_data = IC_pop.get_data("spikes")

        # AC_data = AC_pop.get_data("spikes")
        # ACinh_data = AC_pop_inh.get_data("spikes")

        # Belt_spikes = Belt_pop.getSpikes()
        # Beltinh_spikes = Belt_pop_inh.getSpikes()
        #
        # Belt2_spikes = Belt2_pop.getSpikes()

        # if en_stdp:
        #     if run_one:
        #         varying_weights.append(weights)
        #         run_one=False
        #     weights = ic2ac_proj.get("weight", "list", with_address=True)
        #     varying_weights.append(weights)

    # End simulation
    sim.end()

else:
    sim.run(sim_duration)
    CH_data = CH_pop.get_data("spikes")
    # # CH_v = CH_pop.get_v()
    # # CH_g = CH_pop.get_gsyn()
    # CHinh_data = CH_pop_inh.get_data("spikes")
    #
    ON_data = ON_pop.get_data("spikes")
    # # ON_v = ON_pop.get_v()
    # # ON_g = ON_pop.get_gsyn()
    #
    # ONinh_data = ON_pop_inh.get_data("spikes")
    # # ONinh_v = ON_pop_inh.get_v()
    # # ONinh_g = ON_pop_inh.get_gsyn()
    #
    PL_data = PL_pop.get_data("spikes")
    # # PL_v = PL_pop.get_v()
    #
    VNTB_data = VNTB_pop.get_data("spikes")
    IC_data = IC_pop.get_data("spikes")
    # AC_data = AC_pop.get_data("spikes")
    # ACinh_data = AC_pop_inh.get_data("spikes")

    sim.end()

num_repeats+=1
dir=results_directory#'../OME_SpiNN/yes_samples/'

numpy.savez_compressed(results_directory+'/brainstem_asc_des_{}s'.format(int(sim_duration/1000.)),
                       ic_times=IC_data.segments[0].spiketrains,onset_times=onset_times,dBSPL=dBSPL)
print "IC spikes saved"
# numpy.save(dir+'/ic_spikes_asc.npy', IC_data.segments[0].spiketrains)
# numpy.save('./ac_spikes.npy', AC_data.segments[0].spiketrains)
#numpy.save('./belt_spikes.npy', Belt_spikes)
#numpy.save('./belt2_spikes.npy', Belt2_spikes)
#raster plots
spike_raster_plot_8(stim_times,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size+1,scale_factor=0.001,title='AN',filepath=dir)
spike_raster_plot_8(CH_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size+1,scale_factor=0.001,title='CN Chopper',filepath=dir)
#cell_voltage_plot(CH_v,plt=plt,duration=sim_duration/1000.,id=760,title='CN Chopper cell id:')
#spike_raster_plot(CHinh_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper inh')

spike_raster_plot_8(ON_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=1+AN_pop_size/10,scale_factor=0.001,title='CN Onset',filepath=dir)
#cell_voltage_plot(ON_v,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset v cell id:')
#cell_voltage_plot(ON_g,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset gsyn cell id:')

#spike_raster_plot(ONinh_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset inh')
#cell_voltage_plot(ONinh_v,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset inhibitory cell id:')
#cell_voltage_plot(ONinh_g,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset inhibitory cell id:')
spike_raster_plot_8(PL_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size+1,scale_factor=0.001,title='CN Primary-like',filepath=dir)
#cell_voltage_plot(PL_v,plt=plt,duration=sim_duration/1000.,id=759,title='CN PL cell id:')
#spike_raster_plot_8(VNTB_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=VNTB_pop_size+1,scale_factor=0.001,title='VNTB')
spike_raster_plot_8(IC_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size+1,scale_factor=0.001,title='IC',filepath=dir)
# spike_raster_plot_8(spike_times_spinn,plt=plt,duration=sim_duration/1000.,ylim=1001,scale_factor=0.001,title='IC')
# spike_raster_plot_8(AC_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AC_pop_size+1 ,scale_factor=0.001,title='AC')
#spike_raster_plot_8(ACinh_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AC_pop_size+1,scale_factor=0.001,title='AC inh')
chosen_int = range(AC_pop_size)#numpy.random.choice(AN_pop_size, 12, replace=False)


# if AC_pop_size <= 100:
#     vary_weight_plot(varying_weights,range(int(AC_pop_size)),chosen_int,sim_duration/1000.,
#                              plt,np=numpy,num_recs=num_repeats,ylim=w_max+(w_max/10.))

#weight_dist_plot(varying_weights,1,plt,w_min,w_max)

#spike_raster_plot_8(Belt_spikes,plt=plt,duration=sim_duration/1000.,ylim=AC_pop_size ,scale_factor=0.001,title='Belt')
#spike_raster_plot(Beltinh_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='Belt inh')

#spike_raster_plot(Belt2_spikes,plt=plt,duration=sim_duration/1000.,ylim=AC_pop_size ,scale_factor=0.001,title='Belt2')

#PSTH plots
if psth:
    print

    # psth_plot_8(plt,numpy.arange(100),spike_times_spinn,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_AN")
    #psth_plot(plt,numpy.arange(400,600),CH_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_CH")
    #psth_plot(plt,numpy.arange(700,800),CHinh_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_CH_inh")
    #psth_plot(plt,numpy.arange(400,600),PL_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_PL")
    #psth_plot_8(plt,numpy.arange(100),IC_data.segments[0].spiketrains,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_IC")
    #psth_plot(plt,numpy.arange(AN_pop_size/10),ON_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_ON")
    #psth_plot_8(plt,range(AC_pop_size),AC_data.segments[0].spiketrains,1,duration=sim_duration/1000.,title="PSTH_AC")
    #psth_plot(plt,numpy.arange(AN_pop_size),Belt_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_Belt")
    #psth_plot(plt,numpy.arange(AN_pop_size),Belt2_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_Belt2")

plt.show()
print ""