import numpy
import math
from scipy.io import wavfile
from scipy.signal import resample

def generate_signal(signal_type="tone",fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05,modulation_freq=0.,
                    modulation_depth=1.,plt=None,file_name=None, silence=True,title=''):
    T = 1./fs
    amp = 1. * 28e-6 * 10. ** (dBSPL / 20.)
    num_samples = numpy.ceil(fs * duration)
    # inverted amp on input gives rarefaction effect for positive pressure (?!))
    if signal_type == "tone":
        signal = [-amp*(numpy.sin(2*numpy.pi*freq*T*i) *
                  (modulation_depth*0.5*(1+numpy.cos(2*numpy.pi*modulation_freq*T*i))))#modulation
                  for i in range(int(num_samples))]

    elif signal_type == "sweep_tone":
        if len(freq)<2:
            print "enter low and high frequency sweep values as freq=[low,high]"
        phi = 0
        f = freq[0]
        delta = 2. * numpy.pi * f * T
        f_delta = (freq[1] - freq[0]) /num_samples
        signal = []
        for i in range(int(num_samples)):
            signal.append(-amp * numpy.sin(phi))
            phi = phi + delta
            f = f + f_delta
            delta = 2. * numpy.pi * f * T

    elif signal_type == "file":
        silence = False
        if file_name:
            [fs_f,signal] = wavfile.read(file_name)
            if signal.shape[1]>1:#stereo
                signal = signal[:, 0]
            fs_f=numpy.float32(fs_f)
            if fs_f != fs:
                secs = len(signal)/fs_f
                num_resamples = secs * fs
                signal = resample(signal,num_resamples)

            signal = numpy.float32(signal)
            max_val=numpy.max(numpy.abs(signal))
            for i in range(len(signal)):
                if signal[i] == max_val:
                    print
                signal[i]/=max_val
                signal[i]*=amp #set loudness
        else:
            raise Exception("must include valid wav filename")

    else:
        print "invalid signal type!"
        signal = []

    # add ramps
    num_ramp_samples = numpy.ceil(fs * ramp_duration)
    step = (numpy.pi / 2.0) / (num_ramp_samples - 1)
    for i in range(int(num_ramp_samples)):
        ramp = (numpy.sin(i * step))
        # on ramp
        signal[i] *= ramp
        # off ramp
        signal[-i] *= ramp
    if silence:
        # add silence
        num_silence_samples = numpy.ceil(fs*silence_duration)
        signal = numpy.concatenate((numpy.zeros(num_silence_samples),signal,numpy.zeros(num_silence_samples)))

    if plt:
        plt.figure(title)
        time = numpy.linspace(0,len(signal)/fs,len(signal))
        plt.plot(time,signal)
        plt.xlabel("time (s)")
        plt.ylabel("signal amplitude")
        plt.xlim((0,len(signal)/fs))
    return numpy.float32(signal)

def generate_psth(target_neuron_ids,spike_trains,bin_width,
                  duration,scale_factor=0.001,Fs=22050.):
    scaled_times=[]
    epochs_per_bin = numpy.round(Fs * bin_width)
    num_bins = numpy.floor((duration*Fs)/epochs_per_bin)
    num_bins = numpy.ceil(duration/bin_width)
    psth = numpy.zeros([len(target_neuron_ids),num_bins])
    #psth = numpy.zeros([len(target_neuron_ids),num_bins])
    psth_row_index=0
    for i in target_neuron_ids:
        #extract target neuron times and scale
        spike_times = [spike_time for (neuron_id, spike_time) in spike_trains if neuron_id==i]
        scaled_times= [spike_time * scale_factor for spike_time in spike_times]
        scaled_times.sort()
        prev_time = 0.
        spike_count = 0
        bins = numpy.arange(bin_width,duration,bin_width)
        for j in scaled_times:
            idx = (numpy.abs(bins - j)).argmin()
            if bins[idx] < j:
                idx+=1
            psth[psth_row_index][idx] += 1

            """if j < (prev_time + bin_width):
                prev_time += j
                spike_count += 1
            else:
                #psth_time_index = j // bin_width
                psth_time_index = prev_time // bin_width
                psth[psth_row_index][psth_time_index] = spike_count
                #reset spike_count and prev_time
                prev_time = j
                spike_count = 1"""
        #increment psth_row_index
        psth_row_index += 1

    sum= numpy.sum(psth,axis=0)
    output = (numpy.sum(psth,axis=0)/numpy.round(Fs * bin_width))/(len(target_neuron_ids)/Fs)
    return output

def spike_raster_plot(spikes,plt,duration,ylim,scale_factor=0.001,title=''):
    if len(spikes) > 0:
        spike_times = [spike_time for (neuron_id, spike_time) in spikes]
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]
        spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
        spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

        ##plot results
        plt.figure(title)
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)

def multi_spike_raster_plot(spikes_list,plt,duration,ylim,scale_factor=0.001,marker_size=3,dopamine_spikes=[],title=''):
    plt.figure(title)
    marker_colours = ['black','blue','red','green']
    marker_styles = ['.','+','*','o']
    count = 0
    for spikes in spikes_list:
        if len(spikes) > 0:
            spike_times = [spike_time for (neuron_id, spike_time) in spikes]
            scaled_times = [spike_time * scale_factor for spike_time in spike_times]
            spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
            spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

            ##plot results
            plt.plot(scaled_times, spike_ids, marker_styles[count], markersize=marker_size,
                     color=marker_colours[count])
            plt.ylim(0, ylim)
            plt.xlim(0, duration)
        count +=1
    if len(dopamine_spikes) > 0:
            spike_times = [spike_time for (neuron_id, spike_time) in dopamine_spikes]
            scaled_times = [spike_time * scale_factor for spike_time in spike_times]
            for xc in scaled_times:
                plt.axvline(x=xc,color='red')
            plt.ylim(0, ylim)
            plt.xlim(0, duration)

def vary_weight_plot(varying_weights,ids,stim_ids,duration,plt,num_recs,np,ylim,title=''):
    varying_weights_array = np.array(varying_weights)
    repeats = np.linspace(0, duration, num_recs)
    sr = math.sqrt(len(ids))
    num_cols = np.ceil(sr)
    num_rows = np.ceil(len(ids)/num_cols)

    plt.figure(title)
    plt.suptitle("weight updates for all connections")

    count=0
    for id in ids:
        plt.subplot(num_rows,num_cols,count+1)
        weights=varying_weights_array[:,count]
        #every number of connections per neuron over time should be equal(no struc plasticity)
        #insane way to get each time element from the weights list
        times = np.zeros((len(weights[0]),len(weights)))
        for i in range(len(weights[0])):
            for j in range(len(weights)):
                times[i,j]=weights[j][i]

        for t in times:
            plt.plot(repeats,t)
        label = plt.ylabel("ID:{}".format(str(id+1)))
        if id in stim_ids:
            label.set_color('red')
        plt.xlim(0,duration)
        plt.ylim(0,ylim)
        count+=1


def cell_voltage_plot(v,plt,duration,scale_factor=0.001,id=0,title=''):
        times = [i[1] for i in v if i[0]==id]
        scaled_times = [time * scale_factor for time in times]
        membrane_voltage = [i[2] for i in v if i[0]==id]
        plt.figure(title + str(id+1))
        plt.plot(scaled_times,membrane_voltage)

#function to create a normal probability distribution of distance connectivity list
def distance_dependent_connectivity(pop_size,weights,delays,min_increment=0,max_increment=1):
    pre_index = 0
    conns = []
    while pre_index < pop_size:
        #
        increment = numpy.unique(
            numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        increment = increment[(increment >= min_increment) & (increment < max_increment)]
        for inc in increment:
            post_index = pre_index + inc
            if post_index < pop_size:
                conns.append((pre_index, post_index, weights[int(inc)],delays[int(inc)]))

        rev_increment = numpy.unique(
            numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        rev_increment = rev_increment[(rev_increment >= min_increment) & (rev_increment < max_increment)]
        for r_inc in rev_increment:
            rev_post_index = pre_index - r_inc
            if rev_post_index >= 0:
                conns.append((pre_index, rev_post_index, weights[int(r_inc)],delays[int(r_inc)]))

        pre_index += 1

    return conns

def fixed_prob_connector(num_pre,num_post,p_connect,weights,delays=1.,self_connections=False,single_pop=False):
    connection_list=[]
    #do pre->post connections
    num_connections_per_neuron = int(num_post*p_connect)
    for id in range(num_pre):
        choice = range(num_post)
        if single_pop:
            #exclude current id from being chosen
            choice.remove(id)
        connections = numpy.random.choice(choice,num_connections_per_neuron,replace=False)
        for target_id in connections:
            connection_list.append((id,target_id,weights,delays))

    if not single_pop:
        # do post->pre connections
        num_connections_per_neuron = num_pre * p_connect
        for id in range(num_post):
            choice = range(num_pre)
            if single_pop:
                # exclude current id from being chosen
                choice.remove(id)
            connections = numpy.random.choice(choice, num_connections_per_neuron, replace=False)
            for target_id in connections:
                connection_list.append((id, target_id, weights, delays))

    return connection_list

def test_filter(audio_data,b0,b1,b2,a0,a1,a2):
    past_input=numpy.zeros(2)
    past_concha=numpy.zeros(2)
    concha=numpy.zeros(len(audio_data))
    for i in range(441,len(audio_data)):
        if i>=1202:
            print ''
        concha[i]=(b0 * audio_data[i]
                  + b1 * audio_data[i-1]#past_input[0]
                  + b2 * audio_data[i-2]#past_input[1]
                  - a1 * concha[i-1]#past_concha[0]
                  - a2 * concha[i-2]#past_concha[1]
                     ) * a0

        #past_input[1] = past_input[0]
        #past_input[0] = audio_data[i]

        #past_concha[1] = past_concha[0]
        #past_concha[0] = concha[i]
    return concha
