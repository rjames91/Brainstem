import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *


Fs = 22050.
audio_data = numpy.load('./audio_data.npy')
duration = len(audio_data)/Fs

plt.figure()
t = numpy.arange(0.0,duration,1/Fs)
plt.plot(t,audio_data)

# Setup pyNN simulation
sim.setup(timestep=1.)
sim.set_number_of_neurons_per_core(sim.extra_models.IZK_curr_exp, 100)
#sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 50)

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
#onset responses
D_stellate_parameters = {
   # 'tau_refrac':    0.7,
    'tau_syn_E':    0.12,#0.1,
    'tau_m':    0.125,#0.3,
    #'e_rev_E':    -70,
    'v_thresh': -53,
    }

D_stellate_parameters = {
   # 'tau_refrac':    0.7,
    ''
    'tau_syn_E':    5.0,#0.12,#0.1,
    'tau_m':   0.15,
#    'v_thresh': -53,
    }

#should cover about 1/3 of AN fibres per octopus cell
fan_in_oct = AN_pop_size//3#100#64#100
oct_pop_size = AN_pop_size//fan_in_oct#AN_pop_size#-fan_in_d+1
oct_pop = sim.Population(oct_pop_size,sim.IF_curr_exp,D_stellate_parameters,label="D_stellate_pop")

#create connectivity list for AN-->D_stellate_pop projections TODO:change this weight to depend on fan_in_d
an_oct_weight = 4#100/fan_in_oct#0.3
an_oct_conns = []
for i in range(oct_pop_size):
    for j in range(fan_in_oct):
        an_index = i*fan_in_oct + j#i+j#
        an_oct_conns.append((an_index,i,an_oct_weight,1.))

#setup projection
an_d_proj = sim.Projection(AN_pop,oct_pop,sim.FromListConnector(an_oct_conns),target="excitatory")
oct_pop.record('spikes')
oct_pop.record_v()

#AN_pop.record('spikes')


T_stellate_parameters_IZH = {'a': 0.02,#Izhikevich
                   'b': 0.2,
                   'c': -65,
                   'd': 10,
                  # 'v_init': -75,
                  # 'u_init': 0,
                   'tau_syn_E': 3,
                   #'tau_syn_I': 10,
                  # 'i_offset': 0
                   }

T_stellate_parameters = {#onset chopper
    'tau_refrac':    0.7,
    'tau_syn_E':    0.7,#0.12,
    'tau_syn_I':    0.1,
    'tau_m':    0.33
    }

T_stellate_parameters = {#sustained chopper
    'tau_m':    3.,#18.,#3.,
    'tau_syn_I': 2.,
    'tau_syn_E': 2.,
    'tau_refrac':    0.8,
}

fan_in_t = 5 #based on 5 fibres innervate each T stellate in vid lec + literature
T_stellate_pop_size = AN_pop_size#AN_pop_size//fan_in_t#-fan_in_t+1
T_stellate_pop = sim.Population(T_stellate_pop_size,sim.IF_curr_exp,T_stellate_parameters,label="T_stellate_pop")

# create connectivity list for AN-->T_stellate_pop projections
an_t_stellate_weight = 4#1.2#4.5#0.25#3.0#0.05#0.025#3. / fan_in_t#0.45/fan_in_t#3.0#
an_t_stellate_conns = []
for i in range(T_stellate_pop_size - fan_in_t):
    for j in range(fan_in_t):
        an_index = i + j#i*fan_in_t+j#
        an_t_stellate_conns.append((an_index, i, an_t_stellate_weight, 1.))
        if i > T_stellate_pop_size - 1 or an_index > AN_pop_size -1:
            raise ValueError('invalid population index!')

# create connectivity list for D-->T_stellate_pop projections
d_t_fan_in = T_stellate_pop_size//oct_pop_size#5 #T_stellate_pop_size - D_stellate_pop_size
d_t_stellate_weight = 0.8#10.0/d_t_fan_in#8.0*an_t_stellate_weight#
d_t_stellate_conns = []
for i in range(oct_pop_size):
    for j in range(d_t_fan_in):
        t_index = i*d_t_fan_in + j#i + j
        d_t_stellate_conns.append((i,t_index,d_t_stellate_weight, 1.))
        if i > oct_pop_size -1 or t_index > T_stellate_pop_size - 1:
            raise ValueError('invalid population index!')

# create connectivity list for T-->T_stellate_pop projections
t_t_stellate_weight = 0.01
max_increment_ex = 10
t_t_delays_ex = numpy.empty(max_increment_ex)
t_t_delays_ex.fill(2.)
#determine distance dependent weight array
t_t_weights = [t_t_stellate_weight/i for i in range(1,max_increment_ex+1)]
t_t_stellate_conns_ex = distance_dependent_connectivity(T_stellate_pop_size,t_t_weights,t_t_delays_ex,
                                                        min_increment=1,max_increment=max_increment_ex)

max_increment_in = 2
t_t_weights_in = numpy.empty(max_increment_in)
t_t_weights_in.fill(0.01)
t_t_delays_in = numpy.empty(max_increment_in)
t_t_delays_in.fill(1.)

t_t_stellate_conns_in = distance_dependent_connectivity(T_stellate_pop_size,t_t_weights_in,t_t_delays_in,
                                                        min_increment=1,max_increment=max_increment_in)

#setup projections
an_t_proj = sim.Projection(AN_pop,T_stellate_pop,sim.FromListConnector(an_t_stellate_conns),target="excitatory")
#t_t_proj_ex = sim.Projection(T_stellate_pop,T_stellate_pop,sim.FromListConnector(t_t_stellate_conns_ex),target="excitatory")
#t_t_proj_in = sim.Projection(T_stellate_pop,T_stellate_pop,sim.FromListConnector(t_t_stellate_conns_in),target="inhibitory")
#t_t_proj = sim.Projection(T_stellate_pop,T_stellate_pop,sim.DistanceDependentProbabilityConnector(d_expression="d<10"),target="excitatory")
#d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.FromListConnector(d_t_stellate_conns),target="inhibitory")
#d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.AllToAllConnector(weights=0.2,delays=1.0),target="inhibitory")
d_t_proj = sim.Projection(oct_pop,T_stellate_pop,sim.AllToAllConnector(weights=8.,delays=1.0),target="inhibitory")
#d_t_proj = sim.Projection(D_stellate_pop,T_stellate_pop,sim.AllToAllConnector(weights=8.,delays=1.0),target="inhibitory")

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
#an_pl_proj = sim.Projection(AN_pop,Primary_like_pop,sim.OneToOneConnector(weights=100.,delays=1.0),target="excitatory")

Primary_like_pop.record('spikes')
#Primary_like_pop.record_v()
T_stellate_pop.record('spikes')
T_stellate_pop.record_v()

# Run simulation
sim.run(sim_duration)

T_stellate_spikes = T_stellate_pop.getSpikes()
T_stellate_v = T_stellate_pop.get_v()
D_stellate_spikes = oct_pop.getSpikes()
D_stellate_v = oct_pop.get_v()
#Primary_like_spikes = Primary_like_pop.getSpikes()
#Primary_like_v = Primary_like_pop.get_v()

# End simulation
sim.end()

spike_raster_plot(T_stellate_spikes,plt=plt,duration=duration,ylim=T_stellate_pop_size,scale_factor=0.001)
spike_raster_plot(D_stellate_spikes,plt=plt,duration=duration,ylim=oct_pop_size+1,scale_factor=0.001)
cell_voltage_plot(T_stellate_v,plt=plt,duration=duration,id=65)
#cell_voltage_plot(D_stellate_v,plt=plt,duration=duration)
#spike_raster_plot(Primary_like_spikes,plt=plt,duration=duration,ylim=Primary_like_pop_size,scale_factor=0.001)
#cell_voltage_plot(Primary_like_v,plt=plt,duration=duration)

PSTH_O = generate_psth(numpy.arange(oct_pop_size),D_stellate_spikes,0.001,
                      duration,scale_factor=0.001,Fs=Fs)
PSTH_T = generate_psth(numpy.arange(60,70),T_stellate_spikes,0.001,
                      duration,scale_factor=0.001,Fs=Fs)
#PSTH_PL = generate_psth(numpy.arange(Primary_like_pop_size),Primary_like_spikes,0.001,
#                     duration,scale_factor=0.001,Fs=Fs)
PSTH_AN = generate_psth(numpy.arange(60,70),spike_trains,0.001,
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