import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *


Fs=22050.
audio_data = numpy.load('./audio_data.npy')
duration = len(audio_data)/Fs
print len(audio_data)

plt.figure()
t = numpy.arange(0.0,duration,1/Fs)
plt.plot(t,audio_data)

# Setup pyNN simulation
sim.setup(timestep=1.)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 100)

#open AN spike source
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_kate_a_10kfib.npy")
spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_6k_640fib_50dB.npy")
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_v2.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = numpy.max(spike_ids)


spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
an_scale_factor = duration/numpy.max(spike_times)
scaled_times = [spike_time * an_scale_factor for spike_time in spike_times]
sim_duration = numpy.max(scaled_times)*1000
plt.figure()
plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
plt.ylim(1, numpy.max(spike_ids) + 1)
plt.xlim(0, duration)

#plt.show()

#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

#for i in range(AN_pop_size):
#        id_times = [1000*an_scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn",spike_times_spinn)

spike_times_spinn=numpy.load("./spike_times_spinn.npy")


#create spike source population
AN_pop=sim.Population(AN_pop_size,sim.SpikeSourceArray,{'spike_times':spike_times_spinn},
                        label='AN pop')

#Create D stellate AVCN population
#this is known as a wide band inhibitor population with
#onset chopper responses
D_stellate_parameters = {
    'tau_refrac':    0.7,
    'tau_syn_E':    0.12,#0.1,
    'tau_m':    0.125,#0.3,
    #'e_rev_E':    -53*8.57,
    'v_thresh': -53,
    }

fan_in_d = AN_pop_size#64#100
D_stellate_pop_size = 1#AN_pop_size#-fan_in_d+1
D_stellate_pop = sim.Population(D_stellate_pop_size,sim.IF_cond_exp,D_stellate_parameters,label="D_stellate_pop")

#create connectivity list for AN-->D_stellate_pop projections
an_d_stellate_weight = 80./fan_in_d#65./fan_in_d#70./fan_in_d
an_d_stellate_conns = []
for i in range(D_stellate_pop_size):
    for j in range(fan_in_d):
        an_index = i+j#i*fan_in + j
        an_d_stellate_conns.append((an_index,i,an_d_stellate_weight,1.))

#setup projection
an_d_proj = sim.Projection(AN_pop,D_stellate_pop,sim.FromListConnector(an_d_stellate_conns),target="excitatory")
D_stellate_pop.record('spikes')
D_stellate_pop.record_v()

#AN_pop.record('spikes')

#Create T stellate AVCN population with
#sustained chopper responses TODO:select appropriate parameters for this cell
T_stellate_parameters = {#onset chopper
    'tau_refrac':    0.7,
    'tau_syn_E':    0.7,#0.12,
    'tau_syn_I':    0.1,
    'tau_m':    0.33
    }

T_stellate_parameters = {#sustained chopper
    'tau_m':    3.,
}

T_stellate_pop_size = AN_pop_size#-fan_in_t+1
T_stellate_pop = sim.Population(T_stellate_pop_size,sim.IF_cond_exp,T_stellate_parameters,label="T_stellate_pop")

# create connectivity list for AN-->T_stellate_pop projections
fan_in_t = 5 #based on 5 fibres innervate each T stellate in vid lec
an_t_stellate_weight = 0.025#3. / fan_in_t#0.45/fan_in_t#3.0#
an_t_stellate_conns = []
for i in range(T_stellate_pop_size - fan_in_t):
    for j in range(fan_in_t):
        an_index = i + j
        an_t_stellate_conns.append((an_index, i, an_t_stellate_weight, 1.))
        if i > T_stellate_pop_size - 1 or an_index > AN_pop_size -1:
            raise ValueError('invalid population index!')


# create connectivity list for D-->T_stellate_pop projections
d_t_fan_in = 5 #T_stellate_pop_size - D_stellate_pop_size
d_t_stellate_weight = 2.0/d_t_fan_in#8.0*an_t_stellate_weight#
d_t_stellate_conns = []
for i in range(T_stellate_pop_size - d_t_fan_in):
    for j in range(d_t_fan_in):
        d_index = i + j
        d_t_stellate_conns.append((d_index,i,d_t_stellate_weight, 1.))
#        if i > T_stellate_pop_size - 1 or d_index > D_stellate_pop_size -1:
#            raise ValueError('invalid population index!')

# create connectivity list for T-->T_stellate_pop projections
t_t_stellate_weight = 0.04#0.05#3.0#8.0*an_t_stellate_weight#

t_t_stellate_conns = []
for i in range(T_stellate_pop_size):
    if i % 2 == 0:
        #even to odd excitatory connection
        t_t_stellate_conns.append((i,i+1,t_t_stellate_weight,1.))
        #odd to even excitatory connection
        t_t_stellate_conns.append((i+1,i,t_t_stellate_weight, 1.))

        if i +1 > T_stellate_pop_size -1:
            raise ValueError('invalid population index!')

#setup projections
an_t_proj = sim.Projection(AN_pop,T_stellate_pop,sim.FromListConnector(an_t_stellate_conns),target="excitatory")
t_t_proj = sim.Projection(T_stellate_pop,T_stellate_pop,sim.FromListConnector(t_t_stellate_conns),target="excitatory")
#d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.FromListConnector(d_t_stellate_conns),target="inhibitory")
#d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.AllToAllConnector(weights=0.5,delays=1.0),target="excitatory")
d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.AllToAllConnector(weights=0.15,delays=1.0),target="inhibitory")

#Create primary like CN neuron population
Primary_like_parameters = {
    #'tau_refrac':    0.8,
    #'tau_syn_E':    0.25,
    'tau_syn_E':    1.2,
    'tau_syn_I':    0.1,
    #'tau_m':    0.33
    'tau_m':    1.
    }
Primary_like_pop_size = AN_pop_size#-fan_in_t+1
Primary_like_pop = sim.Population(Primary_like_pop_size,sim.IF_curr_exp,Primary_like_parameters,label="Primary_like_pop")
#setup projections
an_pl_proj = sim.Projection(AN_pop,Primary_like_pop,sim.OneToOneConnector(weights=100.,delays=1.0),target="excitatory")

Primary_like_pop.record('spikes')
#Primary_like_pop.record_v()
T_stellate_pop.record('spikes')
T_stellate_pop.record_v()

# Run simulation
sim.run(sim_duration)

T_stellate_spikes = T_stellate_pop.getSpikes()
T_stellate_v = T_stellate_pop.get_v()
D_stellate_spikes = D_stellate_pop.getSpikes()
D_stellate_v = D_stellate_pop.get_v()
Primary_like_spikes = Primary_like_pop.getSpikes()
#Primary_like_v = Primary_like_pop.get_v()

# End simulation
sim.end()

spike_raster_plot(T_stellate_spikes,plt=plt,duration=duration,ylim=T_stellate_pop_size,scale_factor=0.001)
#spike_raster_plot(D_stellate_spikes,plt=plt,duration=duration,ylim=D_stellate_pop_size,scale_factor=0.001)
#cell_voltage_plot(T_stellate_v,plt=plt,duration=duration)
#cell_voltage_plot(D_stellate_v,plt=plt,duration=duration)
#spike_raster_plot(Primary_like_spikes,plt=plt,duration=duration,ylim=Primary_like_pop_size,scale_factor=0.001)
#cell_voltage_plot(Primary_like_v,plt=plt,duration=duration)

PSTH_O = generate_psth(numpy.arange(D_stellate_pop_size),D_stellate_spikes,0.001,
                      duration,scale_factor=0.001,Fs=Fs)
PSTH_T = generate_psth(numpy.arange(200,500),T_stellate_spikes,0.001,
                      duration,scale_factor=0.001,Fs=Fs)
PSTH_PL = generate_psth(numpy.arange(Primary_like_pop_size),Primary_like_spikes,0.001,
                     duration,scale_factor=0.001,Fs=Fs)
PSTH_AN = generate_psth(numpy.arange(200,500),spike_trains,0.001,
                     duration,scale_factor=an_scale_factor,Fs=Fs)

x = numpy.arange(0,duration,duration/float(len(PSTH_AN)))
#plt.figure()
#plt.plot(x,PSTH_O)
plt.figure()
plt.plot(x,PSTH_T)
plt.figure()
plt.plot(x,PSTH_AN)
#plt.figure()
#plt.plot(x,PSTH_PL)
plt.show()