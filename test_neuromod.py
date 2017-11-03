import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *



# Setup pyNN simulation
sim.setup(timestep=1.)
num_repeats = 20
simulation_time = 1000. * num_repeats

#open AN spike source
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_kate_a_10kfib.npy")
spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_6k_640fib_50dB.npy")
#spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains_v2.npy")
spike_ids = [neuron_id for (neuron_id, spike_time) in spike_trains]
spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]
AN_pop_size = numpy.max(spike_ids)

#rl_pop_size = AN_pop_size**2
rl_pop_size = 1000

#create cortical populations
IZK_regular_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8,
    'v_init': -75,
    'u_init': 0,
    'tau_syn_E': 2,
    'tau_syn_I': 2,
    'i_offset': 0
}
reg_pop_size = int(0.8*rl_pop_size)
cortical_pop_reg = sim.Population(reg_pop_size,sim.extra_models.IZK_curr_exp,IZK_regular_params,label = "cortical_pop_reg")

IZK_fast_params = {
    'a': 0.1,
    'b': 0.2,
    'c': -65,
    'd': 2,
    'v_init': -75,
    'u_init': 0,
    'tau_syn_E': 2,
    'tau_syn_I': 2,
    'i_offset': 0
}
fast_pop_size = int(0.2*rl_pop_size)
cortical_pop_fast = sim.Population(fast_pop_size,sim.extra_models.IZK_curr_exp,IZK_fast_params,label = "cortical_pop_fast")

#create random 10% connectivity between corticial neurons

#create 1Hz poisson spike source to all cortical neurons
poisson_source = sim.Population(rl_pop_size,sim.SpikeSourcePoisson,
                                {'rate': 1.,
                                   'start': 0.,
                                   'duration': simulation_time
                                   })
#connect poisson source to rl pops
possion_proj_reg = sim.Projection(poisson_source,cortical_pop_reg,sim.OneToOneConnector(weights=1.,delays=1.))
possion_proj_fast = sim.Projection(poisson_source,cortical_pop_fast,sim.OneToOneConnector(weights=1.,delays=1.))

#randomly allocate 50 unique to group neuron Ids to 10 groups
S_groups = []
id_samples = numpy.arange(1, rl_pop_size)
for i in range(50):
    S_groups.append(numpy.random.choice(id_samples,5,replace=False))
    id_samples = [id for id in id_samples if id not in S_groups[i]]

#create spike source population


#generate spike times inputs for simulation
spike_times = []
inter_pop_timing_step = 200.
num_sequence_pops = 3

#for i in range(num_sequence_pops):
    #assign additional fixed order firing times
#    for j in range(num_repeats):
#        spike_times.append(j*1000.+400.+i*inter_pop_timing_step)

print "done"

#create connectivity list between spike source population and target neurons of cortical pop

#spike source to cortical pop projection

#create 1Hz poisson source connections to all s_group neurons


#plot input spikes

#setup brainstem populations TODO: change test_brainstem into a class that can be instantiated here

#connect brainstem populations to cortical network

#excitatory ascending projections

#excitatory descending projections

#inhibitory descending projections

cortical_pop_reg.record('spikes')
cortical_pop_fast.record('spikes')

sim.run(simulation_time)

reg_spikes = cortical_pop_reg.getSpikes()
fast_spikes = cortical_pop_fast.getSpikes()

sim.end()

spike_raster_plot(reg_spikes,plt=plt,duration=simulation_time,ylim=reg_pop_size)
spike_raster_plot(fast_spikes,plt=plt,duration=simulation_time,ylim=fast_pop_size)

