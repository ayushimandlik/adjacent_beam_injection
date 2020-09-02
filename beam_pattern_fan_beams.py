#!/home/observer/miniconda2/bin/python
import numpy as N
#import matplotlib.pyplot as M
import argparse, sys

def get_telescope_model(mo_mo):
    f=open(mo_mo)
    lines=f.readlines()[:-4]        #last 4 lines in mo_mo are crap
    modules=[]
    distances=[]
    weights=[]
    for i,line in enumerate(lines):
        l=line.strip().split()
        m, d, w=l[0], float(l[1]), float(l[3])
        modules.append(m)
        if m.startswith("E"):       #taking East as negative direction in distances from center
            d*=-1
        distances.append(d)
        weights.append(w)
    
    modules=N.asarray(modules)
    distances=N.asarray(distances)
    weights=N.asarray(weights)

    order = N.argsort(distances)
    return (modules[order], distances[order], weights[order])

def simulate_telescope(gap):
    #gap=0.1
    wd=778./176
    W=N.arange(0, wd*176, wd) + wd/2 + gap/2
    E=W*-1
    T=N.concatenate([E[::-1], W])
    return T

def false_model():
    w=778./176
    w=4.42
    d=N.arange(-176, 176, 1)* w
    return d

def gen_theta(end, steps):
    tmp=N.arange(0, end, steps)
    theta=N.union1d(tmp[::-1]*-1, tmp)
    theta=N.deg2rad(theta)
    return theta

def get_fbfraction(answer, xx):
  if args.fbs > 0.05:
    raise ValueError("Fan-beam spacing is too large for primary beams effects to be neglected\
        This code cannot incorporate primary beam effects yet. Sorry")

  assert len(args.snrs) ==3, "Need 3 snr values for 3 adjacent beams"
  snrs  = N.array(args.snrs)
  ratio_left = snrs[0]*1./snrs[1]
  ratio_right = snrs[2]*1./snrs[1]

  if (ratio_right >1) or (ratio_left > 1):
    raise ValueError("SNR in the central beam has to be the maximum as a rule")
  
  side, ratio = (-1, ratio_left) if ratio_left >= ratio_right else (1, ratio_right)
  fbs_bins = int(args.fbs/args.res) #fan-beam sep in bins of answer
  if fbs_bins < 10:
    raise RuntimeError("Resolution is too small for the given fb spacing")
  beam_center = N.where(xx==0.0)[0][0]
  beam_range = [beam_center, beam_center + (side * fbs_bins/2)]
  beam_range = sorted(beam_range)
  
  #use = answer[beam_range[0]:beam_range[1]+1]
  use_center = answer[beam_range[0]:beam_range[1]+1]
  use_adjacent = answer[beam_range[0]-side*fbs_bins : beam_range[1]+1-side*fbs_bins]
  ratios = use_adjacent*1./use_center

  #print use_center
  #print use_adjacent
  #ratios = use / use[::-1]
  #print ratios, ratio
  idx = (N.abs(ratios-ratio)).argmin()
  deg_offset_idx = idx + beam_range[0]
  fraction_offset = (deg_offset_idx - beam_center) *1./fbs_bins
  sys.stderr.write("\nFB spacing fractional offset = {}\n".format(fraction_offset))
  sys.stderr.write("Please note:\n\
      This offset is as a fraction of the\n\
      fan-beam-spacing and not as a fraction of the fan-beam-width\n\
      So please make sure that you have given the right 'fbs' in args\n")
  return xx[deg_offset_idx]

def main(args):

    mo_mo = "/home/dada/linux_64/share/molonglo_modules.txt"
    if args.wtfile:
      if args.wtfile.lower()=="current":
	        args.wtfile = mo_mo
      else:
	        mo_mo=args.wtfile
	
    ftop    = 851.220703125 * 1e6 #MHz
    fbottom = 819.970703125 * 1e6 #MHz
    fcenter = (ftop + fbottom)/2
    C       = 299792458           #ask Einstien why this value
    
    freqs= [fbottom, fcenter, ftop]
    if args.f:
        f=N.asarray(args.f)
        freqs=f * 1e6

    freqs=N.asarray(freqs)
        
    k=1**0.5                                    #this absoulte value does not matter
    vals=N.ones(352)*k + 1j*N.ones(352)*k
    if args.sim:
        dists=simulate_telescope(args.gap)
    else:
        mods, dists, weights = get_telescope_model(mo_mo)
    #dists=false_model()
    
    A, angle= 1, 0#N.pi/4                         #these values don't matter at all
    if args.wtfile:
        A*=weights
        print("Module weights applied")
    vals= A * N.exp(1j * angle)
    
    theta=gen_theta(args.range, args.res)

    wavelength = C/freqs
    
    for z,l in enumerate(wavelength):
        ph_ofs=[]                               #Phase_offsets we need to apply per antenna for each theta. Thus it becomes a 2D array
        for t in theta:
            ph_ofs.append(dists*N.sin(t)/l * 2*N.pi)    #d * sin(theta) / lambda * 2*pi = phase_offset for that theta
        
        ans=[]
        for p in ph_ofs:
            ans.append(vals * N.exp(1j*p))      #Here we apply the phase offsets to all modules
        
        ans=N.asarray(ans)
        xx=N.rad2deg(theta)

        answer = N.abs(ans.sum(axis=1))   #Sum of voltages
        peak = N.max(answer)              #Getting the maxima of amplitudes
        answer *= answer                  #Converting the amplitude to power
        answer /= N.max(answer)           #Normalizing to 1.0
        if args.snrs:
          fractional_fb_idx = get_fbfraction(answer, xx)
        #answer *= peak                    #Normalizing so that the peak is the same as the effective number of active modules (sum of amplitudes/weights)
        print(xx, answer)

#        M.plot(xx, answer, label="freq="+str(freqs[z]*1e-6)+"MHz")
#        if args.snrs:
#          M.axvline(fractional_fb_idx, ls='--', c='k')
#    M.legend(loc=1)
#    M.xlabel("Degrees from zenith\nPositive towards west")
#    M.ylabel("Total power in a beam at that angle")
#    M.title("Grating lobes for a fan beam pointing at zenith")
#    M.show()


if __name__ == '__main__':
    a=argparse.ArgumentParser()
    a.add_argument("-f", type = float,  nargs='+', help="List of frequency channels to plot for, default = ftop, fcenter, fbottom of our band", default=None)
    a.add_argument("-wtfile", type=str, help="Apply module weights? If yes, give the path to molonglo_modules.txt from which to pick up weights. Say 'CURRENT' for picking up the latest weights", default=None)
    a.add_argument("-sim", type=bool, help="type True for simulating the array, instead of reading in the actual values; default=False", default=False)
    a.add_argument("-gap", type=float, help="Gap in metres if simulating, default=15", default=15.0)
    a.add_argument("-range", type=float, help="Angle from zenith(in degrees) upto which grating lobes have to be plotted; default = 5", default=5.0)
    a.add_argument("-res", type=float, help="Resolution in degrees required for the plot; default=0.001", default=0.001)
    a.add_argument("-snrs", type=float, nargs='+', help="SNRs in adjacent beams sepearated by space, in the order [left, center, right]. Give zeros where the SNR is unknown", default=None)
    a.add_argument("-fbs", type=float, help="Fan-beam spacing in degrees, only required if giving snrs (def = 4.0/351)", default=(4.0/351))
    args=a.parse_args()

    main(args)
