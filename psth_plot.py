import numpy
import pylab as plt
from signal_prep import *

Fs = 22050.
an_scale_factor = 1./Fs
seg_size = 96.
audio_data = generate_signal(freq=6900,dBSPL=68.,duration=0.4,
                             modulation_freq=0.,fs=Fs,ramp_duration=0.01,plt=None,silence=True)
audio_data = audio_data[0:int(numpy.floor(len(audio_data)/seg_size)*seg_size)]
duration = len(audio_data)/Fs

spike_trains=numpy.load("/home/rjames/Dropbox (The University of Manchester)/EarProject/spike_trains.npy")

PSTH_AN = generate_psth(numpy.arange(800,900),spike_trains,0.001,
                     duration,scale_factor=an_scale_factor,Fs=Fs)

x = numpy.arange(0,duration,duration/float(len(PSTH_AN)))
plt.figure('PSTH_AN')
plt.plot(x,PSTH_AN)
plt.show()