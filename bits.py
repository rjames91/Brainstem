import numpy as np
from scipy.io import wavfile

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
output = []
for sample in test:
    pair= int16_to_uint8_pair(np.int16(sample))
    unpacked = np.unpackbits(pair)
    packed = np.packbits(unpacked)
    output.append(uint8_pair_to_int16(packed))

output = np.asarray(output)
sub = test - output
print output

