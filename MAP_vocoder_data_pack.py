import numpy as np
from scipy.io import wavfile
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv

# Open the results
n_fibres = 3000
results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
results_file = "/cn_timit_1s_{}an_fibres_0.1ms_timestep_50dB_lateralTrue.npz".format(n_fibres)

results_data = np.load(results_directory+results_file)
moc_att = results_data['moc_att']
an_spikes = results_data['an_spikes']
wav_directory = '../OME_SpiNN/'
dBSPL = 50.
Fs = 50e3

timit_l = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788.wav',plt=None,channel=0)
[_,signal] = wavfile.read(wav_directory+'10788.wav')
signal = signal[:,0]
max_val = numpy.max(numpy.abs(signal))
timit_r = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788.wav',plt=None,channel=1,max_val=max_val)
timit = numpy.asarray([timit_l,timit_r])

results_dict = {}
results_dict['audio_stimulus']=timit
results_dict['dBSPL']=dBSPL
results_dict['Fs']=Fs
results_dict['audio_stimulus']=timit
results_dict['an_spikes']=an_spikes
results_dict['moc_att']=moc_att


savemat(results_directory+'/{}_fibres_results'.format(n_fibres),results_dict)