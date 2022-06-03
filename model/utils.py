'''
@File    :   utils.py
@Author  :   Raphael R. Eguchi
@Modified:   
@Contact :   possu@stanford.edu
@License :   (C)Copyright 2022, Raphael R. Eguchi, Stanford University.
@Desc    :   utilities for the IgVAE.
'''

import torch
import numpy as np

def coords_to_dist(coord):
    # Input: B x n x 3.
    n = coord.size(1)
    G = torch.bmm(coord, coord.transpose(1, 2))
    Gt =  torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :]
    Gt = Gt.repeat(1, n, 1)
    dm = Gt + Gt.transpose(1, 2) - 2*G
    dm = torch.sqrt(dm)[:, None, :, :]
    return dm

def save_pdb(xyz, pdb_out="out.pdb"):
    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()
