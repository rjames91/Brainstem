import xml.etree.ElementTree
import os
from signal_prep import normal_dist_connection_builder
from pyNN.random import RandomDistribution
import numpy as np
import matplotlib.pylab as plt

n_pre = 5000
n_post = n_pre*2/3
mv_size = 255
# conn_list_5k = normal_dist_connection_builder(n_pre,n_post,RandomDistribution,conn_num=60,dist=1.,sigma=n_pre/12.,conn_weight=0.1)
conn_lists = np.load('./conn_lists_{}.npz'.format(n_pre))
conn_list_5k = conn_lists['an_d_list']
pre_index = n_pre/2
post_index = 0#n_post/2
# n_mv_connections = np.sort([pre for (pre,post,w,d) in conn_list if pre>pre_index and pre<pre_index+mv_size and post>post_index and post<post_index+mv_size])
mv2mv_connections_5k = []
for pre_index in range(0,n_pre,mv_size):
    n_mv_conns = len([pre for (pre,post,w,d) in conn_list_5k if pre>pre_index and pre<pre_index+mv_size and post>post_index and post<post_index+mv_size])
    mv2mv_connections_5k.append(n_mv_conns)

# n_pre = 10000
# n_post = n_pre*2/3
# conn_list_10k = normal_dist_connection_builder(n_pre,n_post,RandomDistribution,conn_num=60,dist=1.,sigma=n_pre/12.,conn_weight=0.1)
# pre_index = n_pre/2
# post_index = 0#n_post/2
# mv2mv_connections_10k = []
# for pre_index in range(0,n_pre,mv_size):
#     n_mv_conns = len([pre for (pre,post,w,d) in conn_list_10k if pre>pre_index and pre<pre_index+mv_size and post>post_index and post<post_index+mv_size])
#     mv2mv_connections_10k.append(n_mv_conns)

print("sum 5k:{}".format(np.sum(mv2mv_connections_5k)))
# print("sum 10k:{}".format(np.sum(mv2mv_connections_10k)))

plt.figure("{},{}".format(n_pre,mv_size))
plt.plot(mv2mv_connections_5k)
# plt.plot(mv2mv_connections_10k)

plt.show()

xml_directory = '/home/rjames/edge_filtering_experiment/'
xml_files=[]
total_pre_synaptic_events = {}
n_failed_pop_searches = {}
for _,dirs,_ in os.walk(xml_directory):
    for dir in dirs:
        file_path = xml_directory+dir
        total_pre_synaptic_events[dir] = 0
        n_failed_pop_searches[dir] = 0
        for file in os.listdir(file_path):
            if file.endswith(".xml"):
                # xml_files.append(str(file))
                e = xml.etree.ElementTree.parse(file_path+'/'+file).getroot()
                for child in e:
                    if child.attrib['name'] == "Total_pre_synaptic_events":
                        total_pre_synaptic_events[dir]+=int(child.text)
                    if child.attrib['name'] == "Number of failed pop table searches":
                        n_failed_pop_searches[dir]+=int(child.text)



print ""