import numpy as np

# stimulus onset times is a list of onset times lists for each stimulus
# expected time window values of 0.5s - 1s
# spike time is the output skies from a population, format: [(neuron_id,spike_time),(...),...]
def neuron_correlation(spike_train,time_window, stimulus_onset_times):
    correlations = []#1st dimension is stimulus class
    stimulus_index = 0
    for stimulus in stimulus_onset_times:
        correlations.append([])
        #loop through each stimulus onset time and check all neuron firing times
        #save neuron index if firing is within time window of stimulus onset
        for time in stimulus:
            for (neuron_id, spike_time) in spike_train:
                if spike_time > time and spike_time <= (time + time_window):
                    correlations[stimulus_index].append(neuron_id)
            #correlations[stimulus_index].append([spike_time for (neuron_id,spike_time) in spike_train if spike_time>time and spike_time<=(time+time_window)]
        stimulus_index+=1
    return correlations

#spike_train = [(0,30),(0,50),(2,35),(10,100),(10,101)]
spike_train = np.load('./belt_spikes.npy')
ids = [id for (id,time) in spike_train]
max_id = max(ids)
counts =np.asarray([np.zeros(max_id+1),np.zeros(max_id+1)])
num_repeats = 10
num_classes = 2
num_samples = 1
stimulus_times=[]
for j in range(num_classes):
    stimulus_times.append([])
    for i in range(num_repeats*num_samples+1):
        if j ==0:
            if i == 0:
                stimulus_times[j].append(140)#130)#
            else:
                if (i % 2) == 0:
                    stimulus_times[j].append(stimulus_times[j][i-1]+3500-1820)
                    #stimulus_times[j].append(stimulus_times[j][i-1]+3090-1810)
                else:
                    stimulus_times[j].append(stimulus_times[j][i-1]+1820-140)
                    #stimulus_times[j].append(stimulus_times[j][i-1]+1810-140)
        else:
            if i == 0:
                stimulus_times[j].append(920)#880)#
            else:
                if (i % 2) == 0:
                    stimulus_times[j].append(stimulus_times[j][i-1]+4280-2600)
                    #stimulus_times[j].append(stimulus_times[j][i-1]+3870-2430)
                else:
                    stimulus_times[j].append(stimulus_times[j][i-1]+2600-920)
                    #stimulus_times[j].append(stimulus_times[j][i-1]+2430-920)

#stimulus_times = [[130,1810,3500,4750],[880,2600,3830,5370]]
stimulus_times = [stimulus_times[0][-2:-1],stimulus_times[1][-2:-1]]
#stimulus_times = [stimulus_times[0][0:5],stimulus_times[1][0:1]]


correlation_ids = neuron_correlation(spike_train,500,stimulus_times)
#print correlation_ids
stimulus_count = 0
for stimulus_corr in correlation_ids:
    for id in stimulus_corr:
        counts[stimulus_count][id]+=1
    stimulus_count+=1

import matplotlib.pyplot as plt
max_count = counts.max()
plt.figure()
plt.plot(counts.T)
N = np.arange(1000)
#plt.bar(N,counts[0])
#plt.bar(N,counts[1])
plt.ylim((0,max_count+1))

plt.figure()
plt.hist(counts[0])
plt.figure()
plt.hist(counts[1])

selective_neuron_ids=[]
for i in range(len(counts)):
    id_count = 0
    selective_neuron_ids.append([])
    for count in counts[i]:
        if count >= 1:
            others = range(len(counts))
            others.remove(i)
            # check neuron doesn't respond to other stimuli
            # ensures neuron response is exclusive to a single class
            exclusive = True
            for j in others:
                if counts[j][id_count]!=0:#==4:#
                    exclusive = False
            if exclusive:
                selective_neuron_ids[i].append(id_count)
        id_count+=1

np.save('./selective_ids.npy',selective_neuron_ids)
for i in range(len(selective_neuron_ids)):
    print selective_neuron_ids[i]

plt.show()
