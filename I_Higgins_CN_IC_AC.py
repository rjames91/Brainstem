import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution

psth = True
en_stdp = True
num_repeats = 15
# Setup pyNN simulation
timestep =1.
sim.setup(timestep=timestep,max_delay=50.,min_delay=timestep)
sim.set_number_of_neurons_per_core(sim.extra_models.IZK_curr_exp, 40)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

#open AN spike source
[spike_trains,duration,Fs]=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = numpy.max(spike_ids)
spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
an_scale_factor = 1./Fs#duration/numpy.max(spike_times)
scaled_times = [spike_time * an_scale_factor for spike_time in spike_times]
#sim_duration = numpy.max(scaled_times)*1000
single_duration= numpy.max(scaled_times)*1000
sim_duration = numpy.max(scaled_times)*1000 * num_repeats
spikes_train_an_ms = [(neuron_id,int(1000*timestep*an_scale_factor*spike_time)) for (neuron_id,spike_time) in spike_trains]
#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

#for i in range(AN_pop_size):
#        id_times = [1000*timestep*an_scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn",spike_times_spinn)

spike_times_spinn=numpy.load("./spike_times_spinn.npy")

index=0
spike_times_spinn_repeats=[]
for neuron in spike_times_spinn:
    spike_times_spinn_repeats.append([])
    for t in neuron:
        for i in range(num_repeats):
            spike_times_spinn_repeats[index].append(t + single_duration * i)
    index+=1



#params taken from I Higgins thesis pg. 121
IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                  # 'v_thresh': 30
                   'v_init': -75,
                   'u_init': 10.,#0.,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                   #'i_offset': -15.
                   }

IZH_EX_COR = {'a': 0.01,
                   'b': 0.2,
                   'c': -65,
                   'd': 8,
                   #'v_thresh': 30
                  # 'v_init': -75,
                  # 'u_init': 0,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                   #'i_offset': 0
                   }

IZH_INH = {'a': 0.02,
                   'b': 0.25,
                   'c': -55,
                   'd': 0.05,
                   }


#create populations
AN_pop=sim.Population(AN_pop_size,sim.SpikeSourceArray,{'spike_times':spike_times_spinn},label='AN pop')
CH_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Chopper")
CH_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_INH,label="CN_Chopper_inh")
PL_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Primary like")
ON_pop = sim.Population(AN_pop_size//10.,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Onset")
ON_pop_inh = sim.Population(AN_pop_size//10.,sim.extra_models.IZK_curr_exp,IZH_INH,label="CN_Onset_inh")
IC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="IC")
AC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_COR,label="AC")
AC_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_INH,label="AC_inh")

#TODO: change all weight parameters to random distributions

#AN --> CH connectivity
an2ch_list = []
an2ch_weight = RandomDistribution('uniform',parameters=[0,20])#10.#25.
an2ch_fan = 5
for post in range(AN_pop_size):
    an2ch = RandomDistribution('normal', (post, 26.), rng=NumpyRNG())
    an_idxs = an2ch.next(n=an2ch_fan)
    pre_check = []
    for pre in an_idxs:
        pre = int(pre)
        if pre>=0 and pre<AN_pop_size:
            if pre not in pre_check:
                weight = an2ch_weight.next(n=1)
                an2ch_list.append((pre,int(post),weight,1.))
            pre_check.append(pre)
an2ch_proj = sim.Projection(AN_pop,CH_pop,sim.FromListConnector(an2ch_list))

#CH --> CH_inh connectivity
ch2chinh_proj = sim.Projection(CH_pop,CH_pop_inh,sim.OneToOneConnector(weights=0.5))
#CH_inh --> CH connectivity
chinh2ch_proj = sim.Projection(CH_pop_inh,CH_pop,sim.AllToAllConnector(weights=0.1),target='inhibitory')

#AN --> ON connectivity
an2on_prob=0.15#0.1#0.6#0.54#
an2on_weight =0.8#0.6#0.3#0.26#
#connect from each AN to any any ON with probability of 0.54
an2on_proj = sim.Projection(AN_pop,ON_pop,sim.FixedProbabilityConnector(p_connect=an2on_prob,
                                                                        weights=an2on_weight,delays=1,
                                                                        allow_self_connections=False))
#ON --> ON_inh connectivity
on2oninh_proj = sim.Projection(ON_pop,ON_pop_inh,sim.OneToOneConnector(weights=15.,delays=1.))
oninh2on_proj = sim.Projection(ON_pop_inh,ON_pop,sim.AllToAllConnector(weights=0.3,delays=1.),target='inhibitory')

#AN --> PL connectivity
an2pl_proj = sim.Projection(AN_pop,PL_pop,sim.OneToOneConnector(weights=40.,delays=1.0))

#CH --> IC connectivity
ch2ic_list = []
ch2ic_weight = RandomDistribution('uniform',parameters=[0,5])#5.#2.#10.#25.
ch2ic_fan = 5
for post in range(AN_pop_size):
    ch2ic = RandomDistribution('normal', (post, 2.), rng=NumpyRNG())
    ch_idxs = ch2ic.next(n=ch2ic_fan)
    pre_check=[]
    for pre in ch_idxs:
        pre = int(pre)
        if pre>=0 and pre<AN_pop_size:
            weight = ch2ic_weight.next(n=1)
            if pre not in pre_check:
                ch2ic_list.append((pre,int(post),weight,1.))
            pre_check.append(pre)
ch2ic_proj = sim.Projection(CH_pop,IC_pop,sim.FromListConnector(ch2ic_list))

#ON --> IC connectivity
on2ic_proj = sim.Projection(ON_pop,IC_pop,sim.AllToAllConnector(0.05))

#PL --> IC connectivity
pl2ic_proj = sim.Projection(PL_pop,IC_pop,sim.OneToOneConnector(weights=20.,delays=10.0))

#IC-->AC connectivity
tau_plus = 15.
tau_minus = 25.
a_plus = 0.005
a_minus = 0.015
w_min = 0.
w_max = 10.

time_dep = sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus)
weight_dep = sim.AdditiveWeightDependence(w_min=w_min, w_max=w_max,
                                          A_plus=a_plus, A_minus=a_minus)
stdp = sim.STDPMechanism(time_dep, weight_dep)
if en_stdp:
    syn_dyn = sim.SynapseDynamics(slow=stdp)
else:
    syn_dyn = None

print("************ ------------ using stdp ------------ ***********")
print("tau_plus = %f\ttau_minus = %f"% (tau_plus, tau_minus))
print("w_min = %f\tw_max = %f\ta_plus = %f\ta_minus = %f" %
      (w_min, w_max, a_plus, a_minus))

ic2ac_delays = RandomDistribution('uniform',parameters=[1,51])
ic2ac_weights = RandomDistribution('uniform',parameters=[0.,5.])

ic2ac_conn = []

for pre in range(AN_pop_size):
    for post in range(AN_pop_size):
        weight = ic2ac_weights.next(n=1)
        delay = int(ic2ac_delays.next(n=1))
        ic2ac_conn.append((pre,post,weight,delay))
#ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.FromListConnector(ic2ac_conn),
#                            target='excitatory',synapse_dynamics=None)#syn_dyn)
#ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.AllToAllConnector(weights= ic2ac_weights,delays=ic2ac_delays),
#                            target='excitatory',synapse_dynamics=None)#syn_dyn)
ic2ac_proj = sim.Projection(IC_pop,AC_pop,sim.FixedProbabilityConnector(p_connect=0.1,weights=ic2ac_weights,#p_connect=0.01
                                                                        delays=ic2ac_delays),
                            target='excitatory',synapse_dynamics=syn_dyn)


#AC --> AC_inh connectivity
ac2acinh_proj = sim.Projection(AC_pop,AC_pop_inh,sim.OneToOneConnector(weights=1.0,delays=1.))
acinh2ac_proj = sim.Projection(AC_pop_inh,AC_pop,sim.AllToAllConnector(weights=0.1,delays=1.),target='inhibitory')


#setup recordings
CH_pop.record('spikes')
CH_pop.record_v()
#CH_pop.record_gsyn()
CH_pop_inh.record('spikes')
ON_pop.record('spikes')
ON_pop.record_v()
#ON_pop.record_gsyn()

ON_pop_inh.record('spikes')
ON_pop_inh.record_v()
#ON_pop_inh.record_gsyn()

PL_pop.record('spikes')
#PL_pop.record_v()
IC_pop.record('spikes')

AC_pop.record('spikes')
AC_pop_inh.record('spikes')

# Run simulation
varying_weights = []
for i in range(num_repeats):

    sim.run(sim_duration/num_repeats)

    CH_spikes = CH_pop.getSpikes()
    CH_v = CH_pop.get_v()
    #CH_g = CH_pop.get_gsyn()
    CHinh_spikes = CH_pop_inh.getSpikes()

    ON_spikes = ON_pop.getSpikes()
    ON_v = ON_pop.get_v()
    #ON_g = ON_pop.get_gsyn()

    ONinh_spikes = ON_pop_inh.getSpikes()
    ONinh_v = ON_pop_inh.get_v()
    #ONinh_g = ON_pop_inh.get_gsyn()

    PL_spikes = PL_pop.getSpikes()
    #PL_v = PL_pop.get_v()

    IC_spikes = IC_pop.getSpikes()

    AC_spikes = AC_pop.getSpikes()
    ACinh_spikes = AC_pop_inh.getSpikes()

    if en_stdp:
        ic2ac_weights = ic2ac_proj.getWeights(format="array")
        group_weights = []
        for id in range(AN_pop_size):  # group_target_ids:
            connection_weights = [weight for weight in ic2ac_weights[id][:] if not math.isnan(weight)]
            group_weights.append(numpy.array(connection_weights))
        varying_weights.append(numpy.array(group_weights))

varying_weights = numpy.array(varying_weights)
# End simulation
sim.end()

numpy.save("./weights.npy",varying_weights)
if num_repeats>1:
    vary_weight_plot(varying_weights,range(AN_pop_size),[],duration,
                         plt,np=numpy,num_recs=num_repeats,ylim=w_max+1)

#raster plots
spike_raster_plot(spikes_train_an_ms,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='AN')
#spike_raster_plot(CH_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper')
#cell_voltage_plot(CH_v,plt=plt,duration=duration,id=760,title='CN Chopper cell id:')
#spike_raster_plot(CHinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper inh')

#spike_raster_plot(ON_spikes,plt=plt,duration=duration,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset')
#cell_voltage_plot(ON_v,plt=plt,duration=duration,id=76,title='CN Onset v cell id:')
#cell_voltage_plot(ON_g,plt=plt,duration=duration,id=76,title='CN Onset gsyn cell id:')

#spike_raster_plot(ONinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset inh')
#cell_voltage_plot(ONinh_v,plt=plt,duration=duration,id=76,title='CN Onset inhibitory cell id:')
#cell_voltage_plot(ONinh_g,plt=plt,duration=duration,id=76,title='CN Onset inhibitory cell id:')
#spike_raster_plot(PL_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Primary-like')
#cell_voltage_plot(PL_v,plt=plt,duration=duration,id=759,title='CN PL cell id:')

#spike_raster_plot(IC_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='IC')

spike_raster_plot(AC_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='AC')
#spike_raster_plot(ACinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='AC inh')


#PSTH plots
if psth:
    print
    #psth_plot(plt,numpy.arange(700,800),spike_trains,bin_width=0.001,duration=duration,scale_factor=an_scale_factor,Fs=Fs,title="PSTH_AN")
    #psth_plot(plt,numpy.arange(700,800),CH_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_CH")
    #psth_plot(plt,numpy.arange(700,800),CHinh_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_CH_inh")
    #psth_plot(plt,numpy.arange(700,800),PL_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_PL")
    #psth_plot(plt,numpy.arange(700,800),IC_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_IC")
    #psth_plot(plt,numpy.arange(AN_pop_size/10),ON_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_ON")
    #psth_plot(plt,numpy.arange(AN_pop_size),AC_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_AC")

plt.show()