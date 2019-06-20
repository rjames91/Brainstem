import pylab as plt
import numpy as np

import spynnaker8 as sim
import sys
sys.path.append("../")
from signal_prep import *
from spinnak_ear.spinnakear import SpiNNakEar,spinnakear_size_calculator
from pacman.model.constraints.partitioner_constraints.max_vertex_atoms_constraint import MaxVertexAtomsConstraint
from elephant.statistics import isi,cv

#================================================================================================
# Simulation parameters
#================================================================================================
# for _ in range(3):
#     moc_spikes.append([t for t in range(0,150,3)])
Fs = 50e3#100000.#
dBSPL=80
wav_directory = '../../OME_SpiNN/'

click = generate_signal(signal_type='click',fs=Fs,dBSPL=dBSPL,duration=0.00012,ramp_duration=0.,plt=None,silence=True,silence_duration=0.2)
click_stereo = np.asarray([click,click])
binaural_audio = click_stereo

duration = (binaural_audio[0].size/Fs)*1000.
#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=0.1)
#================================================================================================
# Populations
#================================================================================================
an_pop_size = 200
spinnakear_pop_left = sim.Population(an_pop_size,SpiNNakEar(audio_input=binaural_audio[0],fs=Fs,n_channels=an_pop_size/10,ear_index=0),label="spinnakear_pop_left")
spinnakear_pop_left.record(['spikes','moc'])
spinnakear_pop_right = sim.Population(an_pop_size,SpiNNakEar(audio_input=binaural_audio[1],fs=Fs,n_channels=an_pop_size/10,ear_index=1),label="spinnakear_pop_right")
spinnakear_pop_right.record(['spikes','moc'])

#================================================================================================
# Projections
#================================================================================================
sim.run(duration*1.1)

ear_left_data = spinnakear_pop_left.get_data()
ear_spikes_left = np.flipud(ear_left_data['spikes'])
# ear_left_data = spinnakear_pop_left.get_data(['debug','moc'])
# ear_debug_left = np.flipud(ear_left_data['debug'])
ear_moc_left = np.flipud(ear_left_data['moc'])

ear_right_data = spinnakear_pop_right.get_data()
ear_spikes_right = np.flipud(ear_right_data['spikes'])
# ear_right_data = spinnakear_pop_right.get_data(['debug','moc'])
# ear_debug_right = np.flipud(ear_right_data['debug'])
ear_moc_right = np.flipud(ear_right_data['moc'])

sim.end()

spike_raster_plot_8(ear_spikes_left, plt, duration / 1000., an_pop_size + 1, 0.001, title="ear pop activity left")
spike_raster_plot_8(ear_spikes_right, plt, duration / 1000., an_pop_size + 1, 0.001, title="ear pop activity right")
# spike_raster_plot_8(output_spikes_right, plt, duration / 1000., an_pop_size + 1, 0.001, title="output pop activity right")

legend_string = [str(i) for i in range(an_pop_size/10)]
# plt.figure("MOC left")
# for moc_signal in ear_moc_left:
#     x = np.linspace(0,duration,len(moc_signal))
#     plt.plot(x,moc_signal)
# plt.xlabel("time (ms)")
# plt.legend(legend_string)
#
# plt.figure("MOC right")
# for moc_signal in ear_moc_right:
#     x = np.linspace(0,duration,len(moc_signal))
#     plt.plot(x,moc_signal)
# plt.xlabel("time (ms)")
# plt.legend(legend_string)

# legend_string = [str(i) for i in range(an_pop_size/10)]
# plt.figure("debug left")
# for moc_signal in ear_debug_left:
#     x = np.linspace(0,duration,len(moc_signal))
#     plt.plot(x,moc_signal)
# plt.xlabel("time (ms)")
# plt.legend(legend_string)
#
# plt.figure("debug right")
# for moc_signal in ear_debug_right:
#     x = np.linspace(0,duration,len(moc_signal))
#     plt.plot(x,moc_signal)
# plt.xlabel("time (ms)")
# plt.legend(legend_string)

plt.figure("signal")
plt.plot(binaural_audio[0])

# np.savez_compressed('./debug', moc_data=[ear_moc_left,ear_moc_right],ear_debug=[ear_debug_left,ear_debug_right],ear_spikes=[],Fs=Fs, stimulus=binaural_audio)

plt.show()