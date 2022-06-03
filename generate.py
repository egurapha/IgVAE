'''
@File    :   generate.py
@Author  :   Raphael R. Eguchi
@Modified:   
@Contact :   possu@stanford.edu
@License :   (C)Copyright 2022, Raphael R. Eguchi, Stanford University.
@Desc    :   Generation script for IgVAE.
'''
import os
os.sys.path.append('./model/')
import sys
from tqdm import tqdm
import models
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils import *


def generate(model, n, outdir='./', var=1.0, device='cpu'):
    # Generate function with adjustable latent sampling variance.
    for i in tqdm(range(1, n+1)):
        z = torch.FloatTensor(1, 1024, 1, 1).normal_(0., var).to(device=device)
        with torch.no_grad():
            gen_coord = model(z, mode='decode').cpu()
            gen_coord = crop(gen_coord)
        out_path = '%s/gen_%s.pdb' %(outdir, str(i).zfill(4))
        save_pdb(gen_coord[0:,:].detach().cpu().numpy().squeeze(), pdb_out=out_path)


def crop(gen_coord, refpath='./model/cropref/'):
    # Crops Generated Structures based on known Ig data in refpath.
    gen_dm = coords_to_dist(gen_coord[:,1::4,:])
    nn_dist = float('inf')
    nn_idx = None
    for root, dirs, files in os.walk(refpath):
        for file in files:
            if file.endswith('.pth') and not file.startswith('.'):
                r_coord, dm_mask, n_res, aidx = torch.load(root+file)
                r_dm = coords_to_dist(r_coord[:,1::4,:])
                batch_dist = torch.sqrt(((r_dm - gen_dm) * dm_mask).pow(2).sum((1,2,3)))/(n_res * (n_res-1))
                min_idx = torch.argmin(batch_dist)
                min_dist = batch_dist[min_idx]
                if min_dist < nn_dist:
                    nn_dist = min_dist
                    crop_coord = gen_coord[:, aidx[min_idx,0]:aidx[min_idx,1], :]
    return crop_coord


if __name__ == '__main__':
    # Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=5, help="Number of Sampled Structures.")
    parser.add_argument("-seed", type=int, default=4, help="Random Seed.")
    parser.add_argument("-var", type=float, default=1.0, help="Sampling Variance.\
         A larger value will yield more diversity but may decrease quality if too high.")
    parser.add_argument("-outdir", type=str, default='outputs/', help="Output Directory.")
    parser.add_argument("-device", type=str, default='cpu', help="'cuda' or 'cpu'")
    args = parser.parse_args()

    # Output Path.
    if not os.path.isdir(args.outdir):
        os.system('mkdir %s' %args.outdir)        

    # Initialize VAE.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    VAE = models.VAE().to(device=args.device)
    VAE.load_state_dict(torch.load('model/weights/weights.pth'))
    VAE.eval()

    # Generate.
    generate(VAE, args.n, outdir=args.outdir, var=args.var, device=args.device) 

