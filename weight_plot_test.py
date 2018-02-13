import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution

sim_duration = 33600.
w_max = 4.

varying_weights = numpy.load("./weights_to_belt.npy")
ids = RandomDistribution('uniform',parameters=[0,999])
chosen=ids.next(n=100)
chosen_int = [int(id) for id in chosen]
#chosen_int = [380]#single target id

#vary_weight_plot(varying_weights,chosen_int,[],sim_duration,
#                         plt,np=numpy,num_recs=20,ylim=w_max+1)

weight_dist_plot(varying_weights,1,plt)

plt.show()