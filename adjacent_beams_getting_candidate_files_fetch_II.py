from datetime import datetime
startTime = datetime.now()
import numpy as np
import os
import sys
import warnings
import argparse
from sigpyproc.Readers import FilReader as F
from Furby_reader import Furby_reader as Fr
from fan_beams_find_power import power
import random
from operator import add
import pandas as pd
from multiprocessing import Pool
from boxcar_snr_tstamp_dm import read_and_fscrunch, add_noise, get_snr

def get_injected_data(furby, filt, samp, power):
    ff = Fr(furby)
    ff_data = ff.read_data()
    filt_data = filt.readBlock(samp, ff.header.NSAMPS)
    rms_of_filt_data_per_chan = filt_data.std(axis=1)
    added = filt_data + (ff_data * rms_of_filt_data_per_chan[:, None] * power)
    return added.astype(filt.header.dtype)


def write_to_filt(data, out):
    if data.dtype != out.dtype:
        warnings.warn('Given data (dtype={0}) will be unasfely cast to the requested dtype={1} before being written out'.format(
            data.dtype, o.dtype), stacklevel=1)
    out.cwrite(data.T.ravel().astype(out.dtype, casting='unsafe'))


def copy_filts(inp, out, start, end, gulp=8192):
    for nsamps, ii, d in inp.readPlan(gulp, start=start, nsamps=end-start, verbose=False):
        write_to_filt(d, out)


def assert_isamp_sep(isamps):
    x = [0]
    x.extend(list(isamps))
    x = np.array(x)
    diff = x[1:] - x[:-1]
    if np.any(diff < 9000):
        raise ValueError(
            'Injection time stamps cannot be less than 9000 samples apart')


def fetch_h5_files(args):
    print(str(datetime.now() - startTime))
    main_beam_filt, side_beam_filt, furbies , isamp, DM_inj, SNR_inj, output_dir, beam, momo = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]
    f_main = F(main_beam_filt)
    f_side = F(side_beam_filt)
    isamps = int(isamp/f_main.header.tsamp)

    name_main =  main_beam_filt.split('/')[8] + '_' + main_beam_filt.split('/')[9]
    name_side = side_beam_filt.split('/')[8] + '_' + side_beam_filt.split('/')[9]

    power_main_beam, power_side_beam = power(beam, momo)
    fur_name = furbies.split('/')[-1]
    
    o_main = f_main.header.prepOutfile(output_dir + '/injected_main_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_main)
    o_side = f_side.header.prepOutfile(output_dir + '/injected_side_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_side)

    copy_filts(inp=f_main, out=o_main, start= isamps-6000, end=isamps)
    copy_filts(inp=f_side, out=o_side, start= isamps-6000, end=isamps)

    injected_data_main_beam = get_injected_data(furby=furbies, filt=f_main, samp=isamps, power=power_main_beam)
    injected_data_side_beam = get_injected_data(furby=furbies, filt=f_side, samp=isamps, power=power_side_beam)
    write_to_filt(data=injected_data_main_beam, out=o_main)
    write_to_filt(data=injected_data_side_beam, out=o_side)

    copy_filts(inp=f_main, out=o_main, start=isamps+9000, end=isamps+24000)
    copy_filts(inp=f_side, out=o_side, start=isamps+9000, end=isamps+24000)

# The following code has been taken from Vivek Gupta's code on obtaining the snr and width on top of Gaussian noise. This can be used if the usage of Heimdall seems diffult:
    boxcars = [1,2,4,8,16,32,64,128,256]
    ff = Fr(furbies)
    nch = ff.header.NCHAN
    rms_noise_per_channel = 1.0
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
    best_width = boxcars[best_trial]
    best_loc = w_locs[best_trial]

    middle_freq = (819.921955 + 851.171875)/2
    tstart = (6000 - (4.14881e6 * DM_inj * ((middle_freq**-2 - 851.171875**-2)  / (327.68000*10**-3))) + 4500) * (327.68000*10**-6)
    ts_hres = (isamps- (4.14881e6 * DM_inj * ((middle_freq**-2 - 851.171875**-2)  / (327.68000*10**-3))) + 4500) * (327.68000*10**-6) 
    final = pd.DataFrame()
    if beam - int(beam) <= 0.5:
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) -1).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) -2, 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) -2).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres,'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) -1, 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank':output_dir + '/injected_main_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_main, 
            'snr': best_snr, 'tstart': tstart, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam), 'main_side': 'main'}, ignore_index = True)
        final = final.append({'filterbank': output_dir + '/injected_side_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_side, 
            'snr': best_snr, 'tstart': tstart, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam)+1, 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) +2).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) +2, 'main_side': 'side'}, ignore_index = True)
    if beam - int(beam) > 0.5:
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) -1).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) -1, 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank': output_dir + '/injected_main_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_main, 
            'snr': best_snr, 'tstart': tstart, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam), 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank': output_dir + '/injected_side_beam_' + str(beam) + '_' + str(isamp) + '_' + fur_name + '_' + name_side, 
            'snr': best_snr, 'tstart': tstart, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) + 1,'main_side': 'main'}, ignore_index = True)
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) +2).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) +2, 'main_side': 'side'}, ignore_index = True)
        final = final.append({'filterbank': values.high_res + '/BEAM_' + str(int(beam) +3).zfill(3) + '/' + values.utc + '.fil', 
            'snr': best_snr, 'tstart': ts_hres, 'dm': DM_inj, 'width': best_width, 'label': 1.0, 'beams': int(beam) +3, 'main_side': 'side'}, ignore_index = True)

    return final

if __name__ == '__main__':
    a = argparse.ArgumentParser()
#If the candidate parameter file is already made:
    a.add_argument('-c', '--cand_param_file', help='csv file with candidate parameters', type=str)
    a.add_argument('-o', '--output_directory', help = 'output directory', type=str, required = True)
    a.add_argument('-f', '--furby_file', help = 'furby catalog file', type = str)
    a.add_argument('-p', '--path_to_furby', help = 'Path to furby template files', default = './', type = str)
    a.add_argument('-r', '--high_res', help = 'Path to the high resolution filterbanks with all beams', default = './', type = str)
    a.add_argument('-u', '--utc', help = 'utc of the observation', default = '2020-05-08-07:33:43', type=str)
    a.add_argument('-b', '--beam_range', help = 'Range of the beams in which furby is to be added', default = [3,350], type=int, nargs='+')
    a.add_argument('-n', '--number', help = 'Number of filterbanks with furbies inserted to be made', default = 2000, type=int)
    a.add_argument('-t', '--time_stamps', help = 'Range of time stamps on which the furby can be added in samples', default = [100,1700], type=int, nargs='+')
    a.add_argument('-npr', '--nprocess', help = 'Number of processoers available', default = 4, type=int)

    print(str(datetime.now() - startTime))

    values = a.parse_args()
    if values.cand_param_file is None:
        values.cand_param_file = values.output_directory + '/furby_beam_info.csv'
        beam = [random.randint(values.beam_range[0], values.beam_range[1]) for _ in range(values.number)]
        side_beam_frac = [random.randint(0, 9) for _ in range(values.number)]
        side_beam_fraction = [i * 0.1 for i in side_beam_frac]
        beams = list( map(add, beam, side_beam_fraction) )
        df = pd.DataFrame()
        furbies = pd.read_csv(values.furby_file, skiprows = [0,1,2], delim_whitespace = True)
        for i in beams:
            main_beam_filt = values.high_res + '/BEAM_' + str(int(i)).zfill(3) + '/' + values.utc + '.fil'
            side_beam_filt = values.high_res + '/BEAM_' + str(int(i) + 1).zfill(3) + '/' + values.utc + '.fil'
            furbies = random.choice([line.strip() for line in open(values.furby_file, 'r').readlines()[4:]])
            isamps = random.randint(values.time_stamps[0], values.time_stamps[1])
            fub = furbies.split('\t')
            furby = values.path_to_furby + '/furby_' + str(int(fub[0])).zfill(4)
            dm = fub[1]
            snr = fub[4]
            df = df.append({'main_beam' : main_beam_filt, 'side_beam' : side_beam_filt, 'furby_id' : fub[0], 'tstamp' : isamps, 'DM_inj' : dm, 'SNR_inj': snr, 'beam': i}, ignore_index = True)
        df.to_csv(values.cand_param_file, index = None)
    cand_pars = pd.read_csv(values.cand_param_file)
    arguments = []
    for index, row in cand_pars.iterrows():
        arguments.append([row['main_beam'], row['side_beam'], values.path_to_furby + '/furby_' + str(int(row['furby_id'])).zfill(4), 
            row['tstamp'], row['DM_inj'], row['SNR_inj'], values.output_directory, row['beam']])
    cands = pd.DataFrame()
    pool = Pool(processes=values.nprocess)
    cands = cands.append(pool.map(fetch_h5_files, arguments), ignore_index = True)
    print(str(datetime.now() - startTime))
    cands.to_csv(values.output_directory + '/' + 'fetch_path_to_make_cands.csv', index = None)

