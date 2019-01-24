# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:31:42 2019

@author: Henry.Campos
(Actually is a merge and modification of software created by other people!)
This program starts by opening a .segy seismic file and converting it into a numpy array.
Then it calculates single-frequency freqs volumes from that numpy array and saves them as 'Piedras{freqs}Hz.npy


"""

import segyio
with segyio.open('../Desktop/PythonClass/data/d_recon_cwt_Piedras_pcfilt2-48.segy') as s:
    c = segyio.cube(s)

import matplotlib.pyplot as plt
plt.imshow(c[100].T, cmap='Greys')

import numpy as np
 
#fL = 0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
#fH = 0.2  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
#b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
#N = int(np.ceil((4 / b)))
#if not N % 2: N += 1  # Make sure that N is odd.
#n = np.arange(N)
 
# Compute a low-pass filter with cutoff frequency fH.
#hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
#hlpf *= np.blackman(N)
#hlpf = hlpf / np.sum(hlpf)
 
# Compute a high-pass filter with cutoff frequency fL.
#hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
#hhpf *= np.blackman(N)
#hhpf = hhpf / np.sum(hhpf)
#hhpf = -hhpf
#hhpf[(N - 1) / 2] += 1
 
# Convolve both filters.
#h = np.convolve(hlpf, hhpf)

#Applying the filter h to the seismic cube c
#cfilt = np.convolve(c, h)

#plt.imshow(cfilt[100].T, cmap='Greys')

from bruges.attribute import spectraldecomp
sub_volume = c[100:200,100:200, 950:]
out = []
print(c.shape)
freqs = np.arange(2, 10, 4)
for freq in freqs:
    for i, section in enumerate(sub_volume):
        d = spectraldecomp(section, f=(freq,freq), window_length=0.064, dt=0.004)
        print('done processing inline: ', i)
        out.append(d)
        tiles=np.array(out)
        tiles=tiles[...,0]
        np.save(f'Piedras{freq}Hz.npy',tiles)

plt.imshow(d[...,0].T, cmap='Reds', alpha=0.3)
#plt.imshow(d[...,1].T, cmap='Greens', alpha=0.3)
#plt.imshow(d[...,2].T, cmap='Blues', alpha=0.3)