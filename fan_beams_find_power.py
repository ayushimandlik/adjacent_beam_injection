from beam_pattern_fan_beams import get_telescope_model, gen_theta, get_fbfraction
import numpy as np
import pandas as pd
import os, sys, warnings, argparse

def power(beam, momo):
#    mo_mo = "/fred/oz002/amandlik/Furby_injection_side_beams/molonglo_modules.txt"
    ftop = 851.220703125 * 1e6
    fbottom = 819.970703125 * 1e6
    freq = (ftop+fbottom)/2
    C = 299792458
    wavelength = C/freq
    k = 1**0.5
    vals = np.ones(352)*k + 1j*np.ones(352)*k
    mods, dists, weights = get_telescope_model(mo_mo)
    A, angle= 1, 0
    A*=weights
    vals= A * np.exp(1j * angle)
    theta=gen_theta(5.0, 0.001)
    l = wavelength
    ph_ofs=[]
    for t in theta:
        ph_ofs.append(dists*np.sin(t)/l * 2*np.pi)
    ans=[]
    for p in ph_ofs:
        ans.append(vals * np.exp(1j*p))
    
    ans = np.asarray(ans)
    xx = np.rad2deg(theta)
    answer = np.abs(ans.sum(axis=1))
    peak = np.max(answer)
    
    answer *= answer
    answer /= np.max(answer)
#    beam = 177.3
    fb_offset = 4.0/351
    main_beam = int(beam)
    side_beam_frac = beam - main_beam
    main_beam_power = side_beam_frac*fb_offset
    side_beam_power = (side_beam_frac -1)*fb_offset
    df = pd.DataFrame({0:xx, 1:answer})
    main_beam_power_snr = df.iloc[(df[0]-main_beam_power).abs().argsort()[:1]].reset_index()[1][0]
    side_beam_power_snr = df.iloc[(df[0]-side_beam_power).abs().argsort()[:1]].reset_index()[1][0]
    
    return main_beam_power_snr, side_beam_power_snr

if __name__ == '__main__':
  a = argparse.ArgumentParser()
  a.add_argument('-b', type=float, help="Beams and associated power")

  args = a.parse_args()
  power(args.b)





