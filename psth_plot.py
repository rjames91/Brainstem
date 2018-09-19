import numpy as np
import pylab as plt
from signal_prep import *
dB = 40#20
duration = 30.#60.

input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
# cochlea_file = np.load(input_directory + '/spinnakear_asc_des_a_i_u{}s_{}dB.npz'.format(int(duration),dB))
# cochlea_file = np.load(input_directory + '/spinnakear_asc_des_a_i_u60s_{}dB.npz'.format(dB))
cochlea_file = np.load(input_directory + '/spinnakear_1kHz_60s_{}dB.npz'.format(dB))
an_spikes = cochlea_file['scaled_times']
# brainstem_file = np.load(input_directory + '/brainstem_asc_des_a_i_u_{}s_{}dB.npz'.format(int(duration),dB))
brainstem_file = np.load(input_directory + '/brainstem_1kHz_{}s_{}dB.npz'.format(int(duration),dB))
ic_spikes = brainstem_file['ic_times']
ch_spikes = brainstem_file['ch_times']
chi_spikes = brainstem_file['chi_times']
pl_spikes = brainstem_file['pl_times']
on_spikes = brainstem_file['on_times']
vntb_spikes = brainstem_file['vntb_times']

# spike_raster_plot_8(an_spikes,plt,duration,an_spikes.size+1,0.001,title="an pop activity")
# spike_raster_plot_8(ch_spikes,plt,duration,ch_spikes.size+1,0.001,title="ch pop activity")
# spike_raster_plot_8(chi_spikes,plt,duration,chi_spikes.size+1,0.001,title="ch inh pop activity")
# spike_raster_plot_8(pl_spikes,plt,duration,ch_spikes.size+1,0.001,title="pl pop activity")
# spike_raster_plot_8(ic_spikes,plt,duration,ic_spikes.size+1,0.001,title="ic pop activity")

psth_plot_8(plt,numpy.arange(550,650),an_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_AN")
psth_plot_8(plt,numpy.arange(550,650),ch_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_CH")
psth_plot_8(plt,numpy.arange(550,650),chi_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_CH_inh")
psth_plot_8(plt,numpy.arange(550,650),pl_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_PL")

# psth_plot_8(plt,numpy.arange(500,600),ic_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_IC")
# psth_plot_8(plt,numpy.arange(500,600),an_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_AN")
# psth_plot_8(plt,numpy.arange(500,600),ch_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_CH")
# psth_plot_8(plt,numpy.arange(500,600),chi_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_CH_inh")
# psth_plot_8(plt,numpy.arange(500,600),pl_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_PL")
# psth_plot_8(plt,numpy.arange(500,600),on_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_ON")
# psth_plot_8(plt,numpy.arange(500,600),vntb_spikes,bin_width=0.005,duration=duration,scale_factor=0.001,title="PSTH_VNTB")

plt.show()