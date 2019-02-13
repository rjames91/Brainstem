import pylab as plt
import numpy as np
from glob import glob
import os

n_entries_threshold = 64#1024
directory = './reports_local/reports/2019-02-13-15-10-13-332440/run_1/routing_tables_generated'
fnames = glob(directory+'routing_table_*.rpt')
output_file = open(directory+"problem_routers.txt",'a+')


for f in fnames:
    with open(f) as report:
        first_line = report.readline()
        n_entries=[int(s) for s in first_line.split() if s.isdigit()][0]
        if n_entries > n_entries_threshold:
            placement=os.path.split(f)[1]
            placement=placement.lstrip('routing_table_')
            placement=placement.rstrip('.rpt')
            output_file.write(placement+' '+str(n_entries)+'\n')
