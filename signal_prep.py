import numpy

def generate_signal(signal_type="tone",Fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05,modulation_freq=0.):
    T = 1./Fs
    if signal_type == "tone":
        amp = 28e-6 * 10. ** (dBSPL / 20.)
        num_samples = numpy.ceil(Fs*duration)
        signal = [amp*(numpy.sin(2*numpy.pi*freq*T*i) *
                  (0.5*(1+numpy.cos(2*numpy.pi*modulation_freq*T*i))))#modulation
                  for i in range(int(num_samples))]
        #add ramps
        num_ramp_samples = numpy.ceil(Fs*ramp_duration)
        step = (numpy.pi/2.0)/(num_ramp_samples-1)
        for i in range(int(num_ramp_samples)):
            ramp = (numpy.sin(i*step))
            #on ramp
            signal[i]*=ramp
            #off ramp
            signal[-i]*=ramp
    else:
        signal=[]

    # add silence
    num_silence_samples = numpy.ceil(Fs*silence_duration)
    signal = numpy.concatenate((numpy.zeros(num_silence_samples),signal,numpy.zeros(num_silence_samples)))

    return numpy.float32(signal)

def generate_psth(target_neuron_ids,spike_trains,bin_width,
                  duration,scale_factor=0.001,Fs=22050):

    scaled_times=[]
    psth = numpy.zeros([len(target_neuron_ids),numpy.ceil(duration/bin_width)])
    psth_row_index=0
    for i in target_neuron_ids:
        #extract target neuron times and scale
        spike_times = [spike_time for (neuron_id, spike_time) in spike_trains if neuron_id==i]
        scaled_times= [spike_time * scale_factor for spike_time in spike_times]

        prev_time = 0.
        spike_count = 0
        for j in scaled_times:
            if j < (prev_time + bin_width):
                prev_time += j
                spike_count += 1
            else:
                psth_time_index = j // bin_width
                psth[psth_row_index][psth_time_index] = spike_count

                #reset spike_count and prev_time
                prev_time = j
                spike_count = 1
        #increment psth_row_index
        psth_row_index += 1

    return (numpy.sum(psth,axis=0)/numpy.round(Fs * bin_width))/(len(target_neuron_ids)/Fs)

def spike_raster_plot(spikes,plt,duration,ylim,scale_factor=0.001):
    if len(spikes) > 0:
        spike_times = [spike_time for (neuron_id, spike_time) in spikes]
        # scale_factor = duration/numpy.max(spike_times)
        #scale_factor = 0.001
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]
        spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
        spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

        ##plot results
        plt.figure()
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(1, ylim)
        plt.xlim(0, duration)

def cell_voltage_plot(v,plt,duration,scale_factor=0.001):
        times = [i[1] for i in v if i[0]==0]
        scaled_times = [time * scale_factor for time in times]
        membrane_voltage = [i[2] for i in v if i[0]==0]
        plt.figure()
        plt.plot(scaled_times,membrane_voltage)

