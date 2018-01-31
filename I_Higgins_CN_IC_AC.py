import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution


Fs = 44100.
seg_size = 96
audio_data = generate_signal(signal_type='file',dBSPL=40.,fs=Fs,ramp_duration=0.01,
                             file_name='../OME_SpiNN/yes.wav',plt=None)
audio_data = audio_data[11800:36500]
audio_data = audio_data[0:int(numpy.floor(len(audio_data)/seg_size)*seg_size)]
duration = len(audio_data)/Fs

# Setup pyNN simulation
sim.setup(timestep=1.)
sim.set_number_of_neurons_per_core(sim.extra_models.IZK_curr_exp, 100)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

#open AN spike source
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_kate_a_10kfib.npy")
spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains.npy")
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_v2.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = numpy.max(spike_ids)

spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
an_scale_factor = 1./Fs#duration/numpy.max(spike_times)
scaled_times = [spike_time * an_scale_factor for spike_time in spike_times]
sim_duration = numpy.max(scaled_times)*1000
plt.figure('AN')
plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
plt.ylim(1, numpy.max(spike_ids) + 1)
plt.xlim(0, duration)

#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

#for i in range(AN_pop_size):
#        id_times = [1000*an_scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn",spike_times_spinn)

spike_times_spinn=numpy.load("./spike_times_spinn.npy")

#params taken from I Higgins thesis pg. 121
IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                  # 'v_thresh': 30
                  # 'v_init': -75,
                  # 'u_init': 0,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                  # 'i_offset': 0
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
                  # 'i_offset': 0
                   }

IZH_INH = {'a': 0.02,
                   'b': 0.25,
                   'c': -55,
                   'd': 0.05,
                   #'v_thresh': 30
                  # 'v_init': -75,
                  # 'u_init': 0,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                  # 'i_offset': 0
                   }

#create populations
AN_pop=sim.Population(AN_pop_size,sim.SpikeSourceArray,{'spike_times':spike_times_spinn},label='AN pop')
CH_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Chopper")
#CH_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,FS_IZH,label="CN_Chopper_inh")

PL_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Primary like")
ON_pop = sim.Population(AN_pop_size/10.,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Onset")
ON_pop_inh = sim.Population(AN_pop_size/10.,sim.extra_models.IZK_curr_exp,IZH_INH,label="CN_Onset_inh")
IC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="IC")
AC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_EX_COR,label="AC")
AC_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,IZH_INH,label="AC_inh")

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
#ch2chinh_proj = sim.Projection(CH_pop,CH_pop_inh,sim.OneToOneConnector(weights =2.))
#CH_inh --> CH connectivity
#chinh2ch_proj = sim.Projection(CH_pop_inh,CH_pop,sim.AllToAllConnector(weights = 1.,allow_self_connections=False),
#                               target='inhibitory')

#AN --> ON connectivity
an2on_prob=0.54#0.1#
an2on_weight = 0.26#26.
#connect from each AN to any any ON with probability of 0.54
an2on_proj = sim.Projection(AN_pop,ON_pop,sim.FixedProbabilityConnector(p_connect=an2on_prob,
                                                                        weights=an2on_weight,delays=1,
                                                                        allow_self_connections=False))
#ON --> ON_inh connectivity
on2oninh_proj = sim.Projection(ON_pop,ON_pop_inh,sim.OneToOneConnector(1.,1))
#oninh2on_proj = sim.Projection(ON_pop_inh,ON_pop,sim.AllToAllConnector(weights = 1.,allow_self_connections=False),target='inhibitory')
oninh2on_list=[]
oninh2on_weight = -0.074
for inh in range(AN_pop_size/10):
    for on in range(AN_pop_size/10):
        oninh2on_list.append((inh,on,oninh2on_weight,1.))

oninh2on_proj = sim.Projection(ON_pop_inh, ON_pop,sim.FromListConnector(oninh2on_list),target='inhibitory')

#AN --> PL connectivity
an2pl_proj = sim.Projection(AN_pop,PL_pop,sim.OneToOneConnector(weights=40.,delays=1.0))

#CH --> IC connectivity
ch2ic_list = []
ch2ic_weight = 2.#RandomDistribution('uniform',parameters=[10,15])#10.#25.
ch2ic_fan = 5
for post in range(AN_pop_size):
    ch2ic = RandomDistribution('normal', (post, 2.), rng=NumpyRNG())
    ch_idxs = ch2ic.next(n=ch2ic_fan)
    pre_check=[]
    for pre in ch_idxs:
        pre = int(pre)
        if pre>=0 and pre<AN_pop_size:
            weight = ch2ic_weight#.next(n=1)
            if pre not in pre_check:
                ch2ic_list.append((pre,int(post),weight,1.))
            pre_check.append(pre)
ch2ic_proj = sim.Projection(CH_pop,IC_pop,sim.FromListConnector(ch2ic_list))

#ON --> IC connectivity
on2ic_proj = sim.Projection(ON_pop,IC_pop,sim.AllToAllConnector(0.1))

#PL --> IC connectivity
pl2ic_proj = sim.Projection(PL_pop,IC_pop,sim.OneToOneConnector(weights=20.,delays=1.0))

#setup recordings
CH_pop.record('spikes')
#CH_pop.record_v()
#CH_pop_inh.record('spikes')
ON_pop.record('spikes')
#ON_pop.record_v()
#ON_pop_inh.record('spikes')
PL_pop.record('spikes')
#PL_pop.record_v()
IC_pop.record('spikes')

# Run simulation
sim.run(sim_duration)

CH_spikes = CH_pop.getSpikes()
#CH_v = CH_pop.get_v()
#CHinh_spikes = CH_pop_inh.getSpikes()

ON_spikes = ON_pop.getSpikes()
#ON_v = ON_pop.get_v()
ONinh_spikes = ON_pop_inh.getSpikes()

PL_spikes = PL_pop.getSpikes()
#PL_v = PL_pop.get_v()

IC_spikes = IC_pop.getSpikes()

# End simulation
sim.end()

#raster plots
spike_raster_plot(CH_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper')
#cell_voltage_plot(CH_v,plt=plt,duration=duration,id=760,title='CN Chopper cell id:')
#spike_raster_plot(CHinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper inh')

spike_raster_plot(ON_spikes,plt=plt,duration=duration,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset')
#cell_voltage_plot(CH_v,plt=plt,duration=duration,id=76,title='CN Onset cell id:')
spike_raster_plot(ONinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size/10,scale_factor=0.001,title='CN Onset inh')

#spike_raster_plot(PL_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Primary-like')
#cell_voltage_plot(PL_v,plt=plt,duration=duration,id=759,title='CN PL cell id:')

#spike_raster_plot(IC_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='IC')

#PSTH plots
#psth_plot(plt,numpy.arange(700,800),spike_trains,bin_width=0.001,duration=duration,scale_factor=an_scale_factor,Fs=Fs,title="PSTH_AN")
#psth_plot(plt,numpy.arange(700,800),CH_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_CH")
#psth_plot(plt,numpy.arange(700,800),PL_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_PL")
#psth_plot(plt,numpy.arange(700,800),IC_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_IC")
#psth_plot(plt,numpy.arange(AN_pop_size/10),ON_spikes,bin_width=0.001,duration=duration,scale_factor=0.001,Fs=Fs,title="PSTH_ON")


plt.show()