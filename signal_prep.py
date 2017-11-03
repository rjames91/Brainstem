import numpy
import math

def generate_signal(signal_type="tone",Fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05,modulation_freq=0.,
                    modulation_depth=1.):
    T = 1./Fs
    if signal_type == "tone":
        amp = 28e-6 * 10. ** (dBSPL / 20.)
        num_samples = numpy.ceil(Fs*duration)
        signal = [amp*(numpy.sin(2*numpy.pi*freq*T*i) *
                  (modulation_depth*0.5*(1+numpy.cos(2*numpy.pi*modulation_freq*T*i))))#modulation
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
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]
        spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
        spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

        ##plot results
        plt.figure()
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)

def multi_spike_raster_plot(spikes_list,plt,duration,ylim,scale_factor=0.001,marker_size=3,dopamine_spikes=[]):
    plt.figure()
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

def vary_weight_plot(varying_weights,ids,duration,plt,num_recs,np,ylim):
    repeats = np.linspace(0, duration, num_recs)
    sr = math.sqrt(len(ids))
    num_cols = int(sr)
    num_rows = len(ids)//num_cols

    plt.figure()
    plt.suptitle("weight updates for connections from all stimulus group neurons")

    count=0
    for id in ids:
        plt.subplot(num_rows,num_cols,count+1)
        weights=varying_weights[:,count]
        #every number of connections per neuron over time should be equal(no struc plasticity)
        #insane way to get each time element from the weights list
        times = np.zeros((len(weights[0]),len(weights)))
        for i in range(len(weights[0])):
            for j in range(len(weights)):
                times[i,j]=weights[j][i]

        for t in times:
            plt.plot(repeats,t)
        plt.ylabel("ID:{}".format(str(id)))
        plt.xlim(0,duration)
        plt.ylim(0,ylim)
        count+=1


def cell_voltage_plot(v,plt,duration,scale_factor=0.001,id=0):
        times = [i[1] for i in v if i[0]==id]
        scaled_times = [time * scale_factor for time in times]
        membrane_voltage = [i[2] for i in v if i[0]==id]
        plt.figure()
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
