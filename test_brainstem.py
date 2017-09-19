import spynnaker7.pyNN as sim
import numpy
import pylab as plt


Fs=22050.
#audio_data=numpy.fromfile("./load1_1kate_22k",dtype='float32')
audio_data=numpy.fromfile("./load1_1_6k_22k",dtype='float32')
duration= len(audio_data)/Fs

# Setup pyNN simulation
sim.setup(timestep=1.)
#sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

#open AN spike source
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_kate_a_10kfib.npy")
spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_6k_640fib.npy")
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_v2.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = numpy.max(spike_ids)


spike_times = [spike_time for (neuron_id, spike_time) in spike_trains]
scale_factor = duration/numpy.max(spike_times)
scaled_times = [spike_time * scale_factor for spike_time in spike_times]
sim_duration = numpy.max(scaled_times)*1000

plt.figure()
plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)

#plt.show()

#create spinnaker compatible spike times list of lists
#collect all spike times per each neuron ID and append to list
spike_times_spinn=[]

#for i in range(AN_pop_size):
#        id_times =  [1000*scale_factor*spike_time for (neuron_id,spike_time) in spike_trains if neuron_id==i]
#        spike_times_spinn.append(id_times)
#numpy.save("./spike_times_spinn",spike_times_spinn)

spike_times_spinn=numpy.load("./spike_times_spinn.npy")


#create spike source population
AN_pop=sim.Population(AN_pop_size,sim.SpikeSourceArray,{'spike_times':spike_times_spinn},
                        label='AN pop')

#create cochlear nucleus population
D_stellate_parameters = {
    'tau_refrac':    0.7,
    'cm':    1.0,
    'tau_syn_E':    0.1,
    'v_rest':    -65.0,
    'tau_syn_I':    5.0,
    'tau_m':    0.125,
    'e_rev_E':    8.57,
    'i_offset':    0.0,
    'e_rev_I':    -70,
    'v_thresh':    -50.0,
    'v_reset':    -65.0
    }

#Create D stellate AVCN population
#this is known as a wide band inhibitor population with
#onset chopper responses
fan_in = 100
D_stellate_pop_size = AN_pop_size-fan_in+1
D_stellate_pop = sim.Population(D_stellate_pop_size,sim.IF_cond_exp,D_stellate_parameters,label="D_stellate_pop")

#create connectivity list for AN-->D_stellate_pop projections
an_d_stellate_weight = 80./fan_in
an_d_stellate_conns = []
for i in range(D_stellate_pop_size):
    for j in range(fan_in):
        an_index = i+j#i*fan_in + j
        an_d_stellate_conns.append((an_index,i,an_d_stellate_weight,1.))

#setup projection
an_cn_proj = sim.Projection(AN_pop,D_stellate_pop,sim.FromListConnector(an_d_stellate_conns),target="excitatory")
#an_cn_proj = sim.Projection(AN_pop,D_stellate_pop,sim.OneToOneConnector(weights=2.4,delays=1.),target="excitatory")


D_stellate_pop.record('spikes')
#AN_pop.record('spikes')

# Run simulation
sim.run(sim_duration)

D_stellate_spikes = D_stellate_pop.getSpikes()
#AN_spikes = AN_pop.getSpikes()

# End simulation
sim.end()

if len(D_stellate_spikes)>0:
    spike_times = [spike_time for (neuron_id, spike_time) in D_stellate_spikes]
    scale_factor = duration/numpy.max(spike_times)
    scaled_times = [spike_time * scale_factor for spike_time in spike_times]
    spike_ids = [neuron_id for (neuron_id, spike_time) in D_stellate_spikes]
    spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

    ##plot results
    plt.figure()
    plt.plot(scaled_times, spike_ids, '.', markersize=3,
                     markerfacecolor='black', markeredgecolor='none',
                     markeredgewidth=0)
    plt.ylim(1,numpy.max(spike_ids)+1)
    plt.xlim(0,numpy.max(scaled_times))

plt.show()