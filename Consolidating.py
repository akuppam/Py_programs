import os
import sys
import pandas as pd
import glob
import numpy as np
# from scipy.sparse import csr_matrix
sys.path.append('I:/13_TRANSCAD_DEMAND_MODEL/03_Utilities/Reading Trancad Matrices in Python/')
from READS_MTX import tcw_mtx


periods = {'am': ('amLOVskm.mtx', 'skim_pm.mtx'),
           'md': ('mdAUTOskm.mtx', 'skim_md.mtx'),
           'pm': ('pmLOVskm.mtx', 'skim_pm.mtx'), 
           'nt': ('ntAUTOskm.mtx', 'skim_pm.mtx')}

matrices = {}
mag_folder = 'C:/Users/pcamargo/Downloads/toMAG_FreightModel/skim/MAG/'
pag_folder = 'C:/Users/pcamargo/Downloads/toMAG_FreightModel/skim/PAG/'
nodes = 6122

for period in periods:
    print period
    mag_matrix, pag_matrix = periods[period]
    
    composite_matrix = np.zeros((nodes, nodes), np.float64)
    
    print '    Importing MAG matrix'
    mag = tcw_mtx()
    mag.load(mag_folder + mag_matrix,  justcores='Time', verbose=False)
    mag_time = mag.matrix['Time']

    for i, k in enumerate(mag.RIndex):
        for j, l in enumerate(mag.CIndex):
            composite_matrix[k, l] = mag_time[i, j]

    print '    Importing PAG matrix'
    pag = tcw_mtx()
    pag.load(pag_folder + pag_matrix,  justcores='time', verbose=False)
    pag_time = pag.matrix['time']

    pag.RIndex = pag.RIndex + 5000
    pag.CIndex = pag.CIndex + 5000

    print '    Merging matrices'
    for i, k in enumerate(pag.RIndex):
        for j, l in enumerate(pag.CIndex):
            composite_matrix[k, l] = pag_time[i, j]
    
    print '    Compositing cross-flow matrix'
    for i in xrange(101, 4000):
        for j in xrange(5000, nodes):
            gateway1 = composite_matrix[i, 2363] + composite_matrix[1105, j]
            gateway2 = composite_matrix[i, 2191] + composite_matrix[1110, j]
            composite_matrix[i, j] = min(gateway1, gateway2)
            
    for i in xrange(101, 4000):
        for j in xrange(5000, nodes):
            gateway1 = composite_matrix[2363, i] + composite_matrix[j, 1105]
            gateway2 = composite_matrix[2191, i] + composite_matrix[j, 1110]
            composite_matrix[j, i] = min(gateway1, gateway2)
    
    composite_matrix[6105:6122, :].fill(0)
    composite_matrix[:, 6105:6122].fill(0)
    composite_matrix[0:100, :].fill(0)
    composite_matrix[:, 0:100].fill(0)
    
    
    matrices[period] = np.copy(composite_matrix)

    
tot_matrix = None
for period in matrices:
    if tot_matrix is None:
        tot_matrix = np.copy(matrices[period])
        titles = period
    else:
        tot_matrix = tot_matrix + matrices[period]    
        titles = titles + ',' + period

        
print '\nSaving output'
output = open('C:/Users/pcamargo/Downloads/toMAG_FreightModel/skim/impedance_matrices.csv', 'w')
print >>output, 'O,D,' + titles
for i in range(nodes):
    for j in range(nodes):
        if tot_matrix[i, j] > 0:
            text = str(i) + ',' + str(j)
            for period in matrices:
                text =  text + ',' + str(matrices[period][i, j])
            print >>output, text
output.flush()
output.close()