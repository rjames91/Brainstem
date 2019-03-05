import numpy as np
from scipy.io import wavfile
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv

# Open the results
n_fibres = 10000
dBSPL = 65.

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
results_file = "/cn_timit_1s_{}an_fibres_0.1ms_timestep_{}dB_lateralTrue_moc_True.npz".format(n_fibres,int(dBSPL))

results_data = np.load(results_directory+results_file)
moc_att = results_data['moc_att']
an_spikes = results_data['an_spikes']
wav_directory = '../OME_SpiNN/'
Fs = 50e3

max_power = min([np.log10(Fs / 2.), 4.25])
BFs = np.logspace(np.log10(30), max_power, int(n_fibres/10.))

timit_l = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=0)
[fs_file,signal] = wavfile.read(wav_directory+'10788_edit.wav')

signal = signal[:,0]

max_val = numpy.max(numpy.abs(signal))
timit_r = generate_signal(signal_type='file',dBSPL=dBSPL,fs=Fs,ramp_duration=0.0025,silence=True,
                            file_name=wav_directory+'10788_edit.wav',plt=None,channel=1,max_val=max_val)
timit = numpy.asarray([timit_l,timit_r])

results_dict = {}
results_dict['audio_stimulus']=timit
results_dict['dBSPL']=dBSPL
results_dict['Fs']=Fs
results_dict['an_spikes']=an_spikes
moc_att_resampled =[]
for i,moc_ear in enumerate(moc_att):
    moc_att_resampled.append([])
    for moc in moc_ear:
        moc_att_resampled[i].append(resample(moc,100.,Fs))

# plt.figure()
# plt.plot(moc_att_resampled[0][0])
# plt.show()
results_dict['moc_att']=moc_att_resampled
results_dict['BFs']=BFs


savemat(results_directory+'/{}_fibres_results'.format(n_fibres),results_dict)