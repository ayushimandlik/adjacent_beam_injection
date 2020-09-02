import numpy as np
import matplotlib.pyplot as plt
import argparse, os, sys
from tqdm import tqdm
from Furby_reader import Furby_reader as F

def read_and_fscrunch(ff):
  '''
  Reads, dedisperses and fscrunches the data in the furby file.
  Input:
  ff- Furby_reader object
  Returns:
  Dedispersed and fscrunched tseries
  '''
  d = ff.read_data(dd=True)
  return d.sum(axis=0)


def convolve_box_car(time_series, width):
  '''
  Convolves the given time series with a box-car of the given width
  '''
  return np.convolve(time_series, np.ones(width), mode='valid')

def get_snr(time_series, width, rms_noise):
  '''
  Get the best snr from the time_serirs for a box-car of given width 
  Returns the snr, area and the sample index of the center (w/2) of the box-car
  which returned the max snr
  '''
  convolution = convolve_box_car(time_series, width)
  loc = np.argmax(convolution)
  area = convolution[loc]

  snr = area / (rms_noise * width**0.5)
  return snr, area, loc + width/2

def add_noise(template, rms_noise):
  '''
  Adds noise to the template.
  Returns noise added time series
  '''
  return template + np.random.normal(0, rms_noise, len(template))

def main():
  rms_noise_per_channel = 1.0
  boxcars = [1,2,4,8,16,32,64,128,256]

  HDR_str = "Furby_file\tHDR_SNR\tHDR_width\tBest_SNR\tBest_width\tBest_loc\tBoxcar_{}\n".format("\tBoxcar_".join(map(str, boxcars)))
  ff = F(furbies)
  nch = ff.header.NCHAN
  rms_noise = rms_noise_per_channel * nch**0.5
  template = read_and_fscrunch(ff)
  time_series = add_noise(template, rms_noise)
  w_snrs = []
  w_areas = []
  w_locs = []
  for boxcar_width in boxcars:
    snr, area, loc = get_snr(time_series, boxcar_width, rms_noise)
    w_snrs.append(snr)
    w_areas.append(area)
    w_locs.append(loc)

  best_trial = np.argmax(w_snrs)
  best_snr = w_snrs[best_trial]
  best_area = w_areas[best_trial]
  best_loc = w_locs[best_trial]
  best_width = boxcars[best_trial]

  ans_str = "{ff}\t{ff_snr:.3f}\t{ff_width:.3f}\t{best_snr:.3f}\t{best_width:.3f}\t{best_loc}\t{all_snrs}\n".format(ff="furby_"+furby_id, ff_snr=ff.header.SNR, ff_width=ff.header.WIDTH, best_snr=best_snr, best_width=best_width * ff.header.TSAMP * 1e-3, best_loc = best_loc, all_snrs= "\t".join(map(str, w_snrs)))


if __name__ == '__main__':
  a = argparse.ArgumentParser()
  a.add_argument("-furby_db", type=str, help='Path to the furby database (def=2019sepDB)', default="/data/mopsr/archives/Furbies/Databases/furby_databse_20k_Aug19/")
  a.add_argument("-outfile", type=str, help='Path to the output file', default="/home/vgupta/Codes/Fake_FRBs/template_box_car_snrs.txt")
  a.add_argument("-min_bw", type=int, help="Minimum box-car width in samples (def=1)", default=1)
  a.add_argument("-max_bw", type=int, help="Maximum box-car width in samples (def=4350)", default=350)

  args=a.parse_args()
  main()

 
