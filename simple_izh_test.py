import spynnaker7.pyNN as sim
import numpy
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution

# Setup pyNN simulation
timestep =1.
sim.setup(timestep=timestep)
sim.set_number_of_neurons_per_core(sim.extra_models.IZK_curr_exp, 40)

IZH_EX_SUBCOR = {'a': 0.02,
                   'b': -0.1,
                   'c': -55,
                   'd': 6,
                   'v_init': -75,
                   'u_init': 10.,
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


spike_times=[[5.,6.,11,12]]

source_pop = sim.Population(1,sim.SpikeSourceArray,{'spike_times':spike_times},label="source")

on_pop = sim.Population(10,sim.extra_models.IZK_curr_exp,IZH_EX_SUBCOR,label="CN_Onset")

inh_pop = sim.Population(10,sim.extra_models.IZK_curr_exp,IZH_INH,label="CN_Onset_inh")

sim.Projection(source_pop,on_pop,sim.AllToAllConnector(30.,1))

sim.Projection(on_pop,inh_pop,sim.OneToOneConnector(10.,1))

sim.Projection(inh_pop,on_pop,sim.AllToAllConnector(50.,1.),target='inhibitory')

on_pop.record('spikes')
on_pop.record_v()
inh_pop.record('spikes')
inh_pop.record_v()

duration = 100.
sim.run(duration)

ON_spikes = on_pop.getSpikes()
ON_v = on_pop.get_v()
INH_spikes = inh_pop.getSpikes()
INH_v = inh_pop.get_v()

sim.end()

spike_raster_plot(ON_spikes,plt=plt,duration=duration/1000,ylim=10.,scale_factor=0.001,title='CN Onset')
cell_voltage_plot(ON_v,plt=plt,duration=duration/1000,id=0,title='ON cell id:')

spike_raster_plot(INH_spikes,plt=plt,duration=duration/1000,ylim=10.,scale_factor=0.001,title='CN Onset inh')
cell_voltage_plot(INH_v,plt=plt,duration=duration/1000,id=0,title='ON inh cell id:')

plt.show()