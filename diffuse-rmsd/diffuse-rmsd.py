import pyemma
import pickle
import numpy as np


#reading matrix
instream = open("../Data/HSP90/RMSDmatrix.pickle", "r")
matrix = pickle.load(instream)
instream.close()

rmsd_entries = np.zeros((len(matrix[0]),len(matrix[0])))
for i in range(len(matrix)):
    for j in range(len(matrix)):
        rmsd_entries[i][j] = matrix[i][j][2]

EPSILON=0.5
K=np.exp(-rmsd_entries**2/EPSILON)
row_sum = np.sum(K, axis=1)
inv_row_sum = 1.0/row_sum
diag_inv_row_sum = np.diag(inv_row_sum)
Q = np.dot(np.dot(diag_inv_row_sum,K),diag_inv_row_sum)
Q_row_sum = np.sum(Q, axis=1)
Q_trans = np.dot(np.diag(1.0/Q_row_sum),Q)
val,vec = np.linalg.eig(Q_trans)
sorted_vals = np.sort(val.real)
indeces = np.argsort(val.real)
sorted_vec = vec[indeces[-5:]]
msm = pyemma.msm.MSM(Q_trans.real)
msm.pcca(5)
msm.metastable_sets
clusters = []
filename = 'cluster_info.dat'
fh = open(filename, 'w')
for i in xrange(len(msm.metastable_sets)):
    for j in xrange(len(msm.metastable_sets[i])):
        fh.write(matrix[msm.metastable_sets[i][j]][0][0]+" "+str(i)+"\n")
        info = (matrix[msm.metastable_sets[i][j]][0][0],i)
        clusters.append(info)
fh.close()
