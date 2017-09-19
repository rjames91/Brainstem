import numpy

def generate_signal(signal_type="tone",Fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05):
    T = 1./Fs
    if signal_type == "tone":
        amp = 28e-6 * 10. ** (dBSPL / 20.)
        num_samples = numpy.ceil(Fs*duration)
        signal = [amp*numpy.sin(2*numpy.pi*freq*T*i) for i in range(int(num_samples))]
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


