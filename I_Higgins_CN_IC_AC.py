import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution


Fs = 22050.
seg_size = 96
audio_data = generate_signal(freq=6900,dBSPL=68.,duration=0.4,
                             modulation_freq=0.,fs=Fs,ramp_duration=0.01,plt=None,silence=True)
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

RS_IZH = {'a': 0.02,#Izhikevich Regular spiking (class 1)
                   'b': 0.2,
                   'c': -65,
                   'd': 8,
                  # 'v_init': -75,
                  # 'u_init': 0,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                  # 'i_offset': 0
                   }
FS_IZH = {'a': 0.1,#Izhikevich Fast spiking (class 2)
                   'b': 0.2,
                   'c': -65,
                   'd': 2,
                  # 'v_init': -75,
                  # 'u_init': 0,
                   #'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                  # 'i_offset': 0
                   }

#create populations
AN_pop=sim.Population(AN_pop_size,sim.SpikeSourceArray,{'spike_times':spike_times_spinn},label='AN pop')
CH_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,RS_IZH,label="CN_Chopper")
CH_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,FS_IZH,label="CN_Chopper_inh")

PL_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,RS_IZH,label="CN_Primary like")
ON_pop = sim.Population(AN_pop_size/10.,sim.extra_models.IZK_curr_exp,RS_IZH,label="CN_Onset")
ON_pop_inh = sim.Population(AN_pop_size/10.,sim.extra_models.IZK_curr_exp,FS_IZH,label="CN_Onset_inh")
IC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,RS_IZH,label="IC")
AC_pop = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,RS_IZH,label="AC")
AC_pop_inh = sim.Population(AN_pop_size,sim.extra_models.IZK_curr_exp,FS_IZH,label="AC_inh")

#AN --> CH connectivity
an2ch_list = []
an2ch_weight = 5.#30.#TODO: change this to uniform dist between 30 and 35
an2ch_fan = 5
for post in range(AN_pop_size):
    an2ch = RandomDistribution('normal', (post, 26.), rng=NumpyRNG())
    ch_idxs = an2ch.next(n=an2ch_fan)
    for pre in ch_idxs:
        if pre>=0 and pre<AN_pop_size:
            an2ch_list.append((int(pre),int(post),an2ch_weight,1.))
an2ch_proj = sim.Projection(AN_pop,CH_pop,sim.FromListConnector(an2ch_list))

#CH --> CH_inh connectivity
ch2chinh_proj = sim.Projection(CH_pop,CH_pop_inh,sim.OneToOneConnector(weights =2.))
#CH_inh --> CH connectivity
chinh2ch_proj = sim.Projection(CH_pop_inh,CH_pop,sim.AllToAllConnector(weights = 0.,allow_self_connections=False),
                               target='inhibitory')

#setup recordings
CH_pop.record('spikes')
CH_pop.record_v()
CH_pop_inh.record('spikes')

# Run simulation
sim.run(sim_duration)

CH_spikes = CH_pop.getSpikes()
CH_v = CH_pop.get_v()

CHinh_spikes = CH_pop_inh.getSpikes()

# End simulation
sim.end()

spike_raster_plot(CH_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper')
cell_voltage_plot(CH_v,plt=plt,duration=duration,id=860,title='CN Chopper cell id:')

spike_raster_plot(CHinh_spikes,plt=plt,duration=duration,ylim=AN_pop_size,scale_factor=0.001,title='CN Chopper inh')

PSTH_AN = generate_psth(numpy.arange(800,900),spike_trains,0.001,
                     duration,scale_factor=an_scale_factor,Fs=Fs)
x = numpy.arange(0,duration,duration/float(len(PSTH_AN)))
plt.figure('PSTH_AN')
plt.plot(x,PSTH_AN)

PSTH_CH = generate_psth(numpy.arange(800,900),CH_spikes,0.001,
                     duration,scale_factor=0.001,Fs=Fs)
x = numpy.arange(0,duration,duration/float(len(PSTH_CH)))
plt.figure('PSTH_CH')
plt.plot(x,PSTH_CH)

plt.show()