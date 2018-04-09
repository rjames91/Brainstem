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
sim.set_number_of_neurons_per_core(sim.Izhikevich, 128)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
#sim.set_number_of_neurons_per_core(sim.SpikeSourceArray,128)

#open AN spike source
#[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_no5.npy")
[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_10sp_2num_5rep.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = 1000#numpy.max(spike_ids)#
AC_pop_size = 100 #* AN_pop_size
spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
an_scale_factor = 1./Fs#duration/numpy.max(spike_times)
scaled_times = [spike_time * an_scale_factor for spike_time in spike_times]
sim_duration =5*1000.#numpy.max(scaled_times)*1000#
num_repeats = int(numpy.ceil(sim_duration/4000))

single_duration= numpy.max(scaled_times)*1000
spikes_train_an_ms = [(neuron_id,int(1000*an_scale_factor*spike_time)) for (neuron_id,spike_time) in spike_trains]

#[spikes_train_an_ms,duration] = numpy.load('./spike_times_an_alt.npy')
#sim_duration = duration * 1000.#600.#
#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

#for i in range(AN_pop_size):
#        id_times = [1000*timestep*an_scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn_10sp_2num_5rep",spike_times_spinn)
test_spikes=[]

spike_times_spinn = numpy.load("./spike_times_spinn_10sp_2num_5rep.npy")#("./spike_times_spinn_an_alt.npy")#("./spike_times_spinn_yes4.npy")
# spike_times_spinn = numpy.load("./ic_spikes.npy")
AN_pop = sim.Population(AN_pop_size, sim.SpikeSourceArray, {'spike_times': spike_times_spinn}, label='AN pop')
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
               'tau_m': 10.0,
               'tau_refrac': 1.0,#2.0,
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

#create populations==================================================================
CH_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Chopper")
CH_pop_inh = sim.Population(AN_pop_size,sim.Izhikevich,IZH_INH,label="CN_Chopper_inh")
PL_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Primary like")
ON_pop = sim.Population(AN_pop_size//10.,sim.Izhikevich,IZH_EX_SUBCOR,label="CN_Onset")
ON_pop_inh = sim.Population(AN_pop_size//10.,sim.Izhikevich,IZH_INH,label="CN_Onset_inh")
IC_pop = sim.Population(AN_pop_size,sim.Izhikevich,IZH_EX_SUBCOR,label="IC")
# AC_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="AC")
# AC_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="AC_inh")
# Belt_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="Belt")
# Belt_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="Belt_inh")
# Belt2_pop = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_EX_COR,label="Belt2")
# Belt2_pop_inh = sim.Population(AC_pop_size ,sim.Izhikevich,IZH_INH,label="Belt2_inh")

AC_pop = sim.Population(AC_pop_size ,sim.IF_curr_exp,LIF_cell_params,label="target")
AC_pop_inh = sim.Population(AC_pop_size ,sim.IF_curr_exp,LIF_cell_params,label="AC_inh")

#AN --> CH connectivity==================================================================
an2ch_weight = RandomDistribution('uniform',(0.5,0.8))#[0.7,0.8])#[0.,2.])#[1.,2.])#10.#25.
#an2ch_fan = 5
diagonal_width = 26.
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AN_pop_size / AN_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
an2ch_list = normal_dist_connection_builder(AN_pop_size,AN_pop_size,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma,an2ch_weight)
an2ch_proj = sim.Projection(AN_pop,CH_pop,sim.FromListConnector(an2ch_list))

#CH --> CH_inh connectivity==================================================================
ch2chinh_proj = sim.Projection(CH_pop,CH_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=0.5))
#CH_inh --> CH connectivity==================================================================
chinh2ch_proj = sim.Projection(CH_pop_inh,CH_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.1),
                               receptor_type='inhibitory')

#AN --> ON connectivity==================================================================
an2on_prob=0.4#0.15#0.6#0.54#
an2on_weight =RandomDistribution('uniform',(0.05,0.15))
an2on_proj = sim.Projection(AN_pop,ON_pop,sim.FixedProbabilityConnector(p_connect=an2on_prob,allow_self_connections=False),
                            synapse_type=sim.StaticSynapse(weight=an2on_weight,delay=1))
#ON --> ON_inh connectivity==================================================================
on2oninh_proj = sim.Projection(ON_pop,ON_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=15.,delay=1.))
oninh2on_proj = sim.Projection(ON_pop_inh,ON_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.3,delay=1.),
                               receptor_type='inhibitory')

#AN --> PL connectivity==================================================================
an2pl_weight =RandomDistribution('uniform',(25.,40.))#[25.,35.])#
an2pl_proj = sim.Projection(AN_pop,PL_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=an2pl_weight,delay=1.0))#(weights=40.,delays=1.0))#

#CH --> IC connectivity==================================================================
ch2ic_weight = RandomDistribution('uniform',(10.,12.))#[4.,5.])
diagonal_width = 2.
diagonal_sparseness = 1.
in2out_sparse = .67 * .67 / diagonal_sparseness
dist = max(int(AN_pop_size / AN_pop_size), 1)
sigma = dist * diagonal_width
conn_num = int(sigma / in2out_sparse)
ch2ic_list = normal_dist_connection_builder(AN_pop_size,AN_pop_size,RandomDistribution,NumpyRNG(),
                                            conn_num,dist,sigma,ch2ic_weight)
ch2ic_proj = sim.Projection(CH_pop,IC_pop,sim.FromListConnector(ch2ic_list))

#ON --> IC connectivity==================================================================
on2ic_weight = RandomDistribution('uniform',(0.01,0.05))#[0,0.1])
on2ic_proj = sim.Projection(ON_pop,IC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=on2ic_weight))

#PL --> IC connectivity==================================================================
pl2ic_weight = RandomDistribution('uniform',(10.,12.))
pl2ic_proj = sim.Projection(PL_pop,IC_pop,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=pl2ic_weight,delay=1.0))

#IC-->AC connectivity==================================================================
tau_factor = 1./timestep
tau_plus = 16#16.7*tau_factor
tau_minus = 30#33.7*tau_factor
a_plus = 0.01#0.005#0.0075
a_minus = 0.05#0.05#0.005#0.005
w2s= 2.#20.
w_min = 0.
w_max = w2s/2#w2s/4.#0.25#2.5#80./50.#40./1000.#0.05#4.#3.
av_weight = w_max/2.#w_max#
ten_perc = av_weight/10.
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
p_connect = 1.0
#create manual fixed prob connector list so I can identify number of connections per AC neuron pre simulation
# ac_connections=numpy.zeros(AC_pop_size)
# ic2ac_list =[]
# for i in range(AN_pop_size):
#     probs = numpy.random.rand(AC_pop_size)
#     present = probs < p_connect
#     ac_connections=numpy.add(ac_connections,present.astype(numpy.int))
#     ids = numpy.where(present)[0]
#     for id in ids:
#         ic2ac_list.append((i,id))
#
# ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.FromListConnector(ic2ac_list),synapse_type=stdp,receptor_type="excitatory")

ic2ac_proj = sim.Projection(
    IC_pop, AC_pop, sim.FixedProbabilityConnector(p_connect=p_connect),
        synapse_type=stdp,receptor_type="excitatory", source=None, space=None)
weights = ic2ac_proj.get("weight", "list", with_address=True)


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
#ac2acinh_proj = sim.Projection(AC_pop,AC_pop_inh,sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=1.0,delay=1.))
#acinh2ac_proj = sim.Projection(AC_pop_inh,AC_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=0.005,delay=1.),
#                               receptor_type='inhibitory')
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
# CH_pop.record('spikes')
# #CH_pop.record_v()
# #CH_pop.record_gsyn()
# CH_pop_inh.record('spikes')
# ON_pop.record('spikes')
# #ON_pop.record_v()
# #ON_pop.record_gsyn()
#
# ON_pop_inh.record('spikes')
# #ON_pop_inh.record_v()
# #ON_pop_inh.record_gsyn()
#
# PL_pop.record('spikes')
# #PL_pop.record_v()
IC_pop.record('spikes')

AC_pop.record('spikes')
#AC_pop_inh.record('spikes')

# Belt_pop.record('spikes')
# Belt_pop_inh.record('spikes')
#
# Belt2_pop.record('spikes')

# Run simulation
if num_repeats>1:
    varying_weights = []
    run_one = True
    # add initial weights
    # varying_weights = [[]]
    # for i in range(AC_pop_size):
    #    varying_weights[0].append(stdp_weights.next(n=ac_connections[i]))

    for i in range(num_repeats):

        sim.run(sim_duration/num_repeats)

        # CH_data = CH_pop.get_data("spikes")
        # # CH_v = CH_pop.get_v()
        # # CH_g = CH_pop.get_gsyn()
        # CHinh_data = CH_pop_inh.get_data("spikes")
        #
        # ON_data = ON_pop.get_data("spikes")
        # # ON_v = ON_pop.get_v()
        # # ON_g = ON_pop.get_gsyn()
        #
        # ONinh_data = ON_pop_inh.get_data("spikes")
        # # ONinh_v = ON_pop_inh.get_v()
        # # ONinh_g = ON_pop_inh.get_gsyn()
        #
        # PL_data = PL_pop.get_data("spikes")
        # # PL_v = PL_pop.get_v()
        #
        IC_data = IC_pop.get_data("spikes")

        AC_data = AC_pop.get_data("spikes")
        #ACinh_data = AC_pop_inh.get_data("spikes")

        # Belt_spikes = Belt_pop.getSpikes()
        # Beltinh_spikes = Belt_pop_inh.getSpikes()
        #
        # Belt2_spikes = Belt2_pop.getSpikes()

        if en_stdp:# and (i==0 or i==(num_repeats-1)):
            if run_one:
                weights_list = [[] for _ in range(AC_pop_size)]
                # add weights for all incoming connections to each AC neuron
                for (pre, post, weight) in weights:
                    weights_list[post].append(weight)
                # weights_list = [weights[i:i+AN_pop_size-1] for i in range(0,len(weights),AN_pop_size)]
                varying_weights.append(weights_list)
                run_one=False
            weights = ic2ac_proj.get("weight", "list", with_address=True)
            #make empty list for each AC neuron
            weights_list = [[]for _ in range(AC_pop_size)]
            #add weights for all incoming connections to each AC neuron
            for (pre, post, weight) in weights:
                weights_list[post].append(weight)
            #weights_list = [weights[i:i+AN_pop_size-1] for i in range(0,len(weights),AN_pop_size)]
            varying_weights.append(weights_list)

    # End simulation
    sim.end()

else:
    sim.run(sim_duration)
    # CH_data = CH_pop.get_data("spikes")
    # # CH_v = CH_pop.get_v()
    # # CH_g = CH_pop.get_gsyn()
    # CHinh_data = CH_pop_inh.get_data("spikes")
    #
    # ON_data = ON_pop.get_data("spikes")
    # # ON_v = ON_pop.get_v()
    # # ON_g = ON_pop.get_gsyn()
    #
    # ONinh_data = ON_pop_inh.get_data("spikes")
    # # ONinh_v = ON_pop_inh.get_v()
    # # ONinh_g = ON_pop_inh.get_gsyn()
    #
    # PL_data = PL_pop.get_data("spikes")
    # # PL_v = PL_pop.get_v()
    #
    IC_data = IC_pop.get_data("spikes")

    AC_data = AC_pop.get_data("spikes")
    #ACinh_data = AC_pop_inh.get_data("spikes")

    sim.end()

# numpy.save('./ic_spikes.npy', IC_data.segments[0].spiketrains)
numpy.save('./ac_spikes.npy', AC_data.segments[0].spiketrains)
#numpy.save('./belt_spikes.npy', Belt_spikes)
#numpy.save('./belt2_spikes.npy', Belt2_spikes)
#raster plots
# spike_raster_plot(spikes_train_an_ms,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='AN')
#spike_raster_plot(CH_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper')
#cell_voltage_plot(CH_v,plt=plt,duration=sim_duration/1000.,id=760,title='CN Chopper cell id:')
#spike_raster_plot(CHinh_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper inh')

#spike_raster_plot(ON_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset')
#cell_voltage_plot(ON_v,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset v cell id:')
#cell_voltage_plot(ON_g,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset gsyn cell id:')

#spike_raster_plot(ONinh_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset inh')
#cell_voltage_plot(ONinh_v,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset inhibitory cell id:')
#cell_voltage_plot(ONinh_g,plt=plt,duration=sim_duration/1000.,id=76,title='CN Onset inhibitory cell id:')
#spike_raster_plot(PL_spikes,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='CN Primary-like')
#cell_voltage_plot(PL_v,plt=plt,duration=sim_duration/1000.,id=759,title='CN PL cell id:')

spike_raster_plot_8(IC_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='IC')
# spike_raster_plot_8(spike_times_spinn,plt=plt,duration=sim_duration/1000.,ylim=1001,scale_factor=0.001,title='IC')
spike_raster_plot_8(AC_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AC_pop_size+1 ,scale_factor=0.001,title='AC')
#spike_raster_plot_8(ACinh_data.segments[0].spiketrains,plt=plt,duration=sim_duration/1000.,ylim=AN_pop_size,scale_factor=0.001,title='AC inh')
chosen_int = range(AC_pop_size)#numpy.random.choice(AN_pop_size, 12, replace=False)

vary_weight_plot(varying_weights,chosen_int,[],sim_duration,
                         plt,np=numpy,num_recs=int(num_repeats+1),ylim=w_max+(w_max/10.))

weight_dist_plot(varying_weights,1,plt,w_min,w_max)


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
    psth_plot_8(plt,range(AC_pop_size),AC_data.segments[0].spiketrains,1,duration=sim_duration/1000.,title="PSTH_AC")
    #psth_plot(plt,numpy.arange(AN_pop_size),Belt_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_Belt")
    #psth_plot(plt,numpy.arange(AN_pop_size),Belt2_spikes,bin_width=0.001,duration=sim_duration/1000.,scale_factor=0.001,title="PSTH_Belt2")

plt.show()
print ""