import numpy

layer_1_size = 10
layer_2_size = 10

weights = numpy.load('./weights.npy')

weights_index=0
connections_list = []
for i in range(layer_1_size):
    for j in range(layer_2_size):
        #tuple structure is (pre_pop_index,post_pop_index,weight,delay)
        connections_list.append((i,j,weights[weights_index],1.))
        weights_index+=1