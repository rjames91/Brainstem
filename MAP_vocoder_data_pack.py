import numpy as np
from scipy.io import wavfile
import pylab as plt
import math
from signal_prep import *
from scipy.io import savemat, loadmat
from elephant.statistics import isi,cv

# Open the results
n_fibres = 1000
dBSPL = 50

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# results_file = "/moc_timit_1s_{}an_fibres_0.1ms_timestep_{}dB_lateralTrue_moc_True.npz".format(n_fibres,int(dBSPL))
results_file = "/cn_timit_0s_{}an_fibres_0.1ms_timestep_{}dB_3s_moc_True_lat_True.npz".format(n_fibres,dBSPL)

results_data = np.load(results_directory+results_file)
an_spikes = [np.flipud(data['spikes']) for data in results_data['ear_data']]
moc_att = [np.flipud(data['moc']) for data in results_data['ear_data']]
wav_directory = '../OME_SpiNN/'
Fs = float(results_data['Fs'])

results_dict = {}
results_dict['audio_stimulus']= results_data['stimulus']
results_dict['dBSPL']=dBSPL
results_dict['Fs']=Fs
results_dict['an_spikes']=an_spikes
max_power = min([np.log10(Fs / 2.), 4.25])
results_dict['BFs'] = np.logspace(np.log10(30), max_power, n_fibres/10)

Fs = 1./100e-6
moc_att_resampled =[]
for i,moc_ear in enumerate(moc_att):
    moc_att_resampled.append([])
    for moc in moc_ear:
        resample_factor = int(Fs / 1000.)
        moc_att_resampled[i].append(moc[::resample_factor])
        # moc_att_resampled[i].append(moc)

#TODO: move resampling to DRNL c code

# plt.figure()
# plt.plot(moc_att_resampled[0][0])
# plt.show()
# results_dict['moc_att']=moc_att
results_dict['moc_att']=moc_att_resampled

savemat(results_directory+'/{}_fibres_results'.format(n_fibres),results_dict)