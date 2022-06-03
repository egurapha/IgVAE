# Ig-VAE
## Author: Raphael R. Eguchi

The Ig-VAE is a variational autoencoder that directly generates the 3D coordinates of immunoglobulin backbones using a torsion- and distance-based loss function that is rotationally and translationally invariant. The model is trained on structures from [AbDb/AbYBank](http://www.abybank.org/abdb/). IgVAE is intended for use with existing protein design suites such as [Rosetta](https://www.rosettacommons.org/software). 

![figure_1_schematic scale=0.1](https://user-images.githubusercontent.com/10354479/171936661-8b2eb74b-b41b-450b-bc67-78536285c3f9.png)

### Requirements
Python 3.7  
See environment.yml

### Usage
```
python generate.py -n 100 -device cpu -seed 14 -outdir outputs/ 
```

### Interpolation Example
<p align="center">
<img src="https://user-images.githubusercontent.com/10354479/171932946-1ad954b3-8f4b-4dd0-8166-c07e4645ccb7.gif" width="300" height="300" />
</p>

### Notes
The environment uses CPU Pytorch, but the script is GPU-compatible via:
```
python generate.py -device cuda
```

### Citing and Licensing
This software is distributed under the BSD-3 license. The license file is available at license.txt.
This work was published in PLoS Computational Biology in 2022. A reference will be made available here soon.
