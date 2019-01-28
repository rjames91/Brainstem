import numpy as np
from scipy.io import wavfile
from scipy import fftpack as fft

def int16_to_uint8_pair(input):
    u_input = np.uint16(input)
    mask = np.uint16(2 ** 8 - 1)
    ls_byte = np.uint8(u_input & mask)
    ms_byte = np.uint8(u_input>>8)
    return [ms_byte,ls_byte]

def uint8_pair_to_int16(input):
    ms_byte_u16 = np.uint16(input[0])
    ls_byte_u16 = np.uint16(input[1])
    combined_u = (ms_byte_u16<<8) | ls_byte_u16
    output = np.int16(combined_u)
    return output

#test = np.arange(1000)
fs,test = wavfile.read('./speech.wav')
frame_length = 1024
num_frames = int(len(test)/frame_length)
test = test[0:frame_length*num_frames]
frames = np.split(test,num_frames)
output = []
for frame in frames:
    #dct
    coeffs = fft.dct(frame,norm='ortho')
    #normalise
    max_coeff = np.amax(abs(coeffs))
    coeffs /= max_coeff
    #scale to byte
    coeffs *= 2.**8 -1
    uint8_coeffs = np.array(np.uint8(coeffs),dtype=np.uint8)
    unpacked = np.unpackbits(uint8_coeffs)
    packed = np.packbits(unpacked)

    #inverse dct
    samples = fft.idct(np.float64(packed),norm='ortho')
    output=np.append(output,samples)

#normalise volume of output
max_sample = max(abs(output))
output /= max_sample


#scale up to 16 bit wav range
output *= 2**15 -1
output = np.int16(output)

#save output wav file
wavfile.write('./speech_decode.wav',fs,output)

#    for sample in test:
        #pair= int16_to_uint8_pair(np.int16(sample))
#        unpacked = np.unpackbits(pair)
#        packed = np.packbits(unpacked)
        #output.append(uint8_pair_to_int16(packed))

#output = np.asarray(output)
#sub = test - output
#print output



