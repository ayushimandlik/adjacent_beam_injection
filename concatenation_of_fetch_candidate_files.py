import pandas as pd
import numpy as np
import glob
import os
import h5py
import sys
from datetime import datetime
import argparse
import locale

def getting_data(h5_file):
    f = h5py.File(h5_file)
    freq_time_data = np.array(f['data_freq_time'])[:, ::-1].T
    dm_time_data = np.array(f['data_dm_time']).T
    return freq_time_data, dm_time_data

def concat(left_2, left_1, main_h5_file, right_1, right_2):
    f = h5py.File(main_h5_file[0])
    dm, cand_id, snr, width, tcand, fch1, foff, tsamp = f.attrs['dm'], f.attrs['cand_id'],  f.attrs['snr'], f.attrs['width'], f.attrs['tcand'], f.attrs['fch1'],  f.attrs['foff'],  f.attrs['tsamp']
    nchan = 320
    dm_opt = f.attrs['dm_opt']
    snr_opt = f.attrs['snr_opt']
    main_ft, main_dmt = getting_data(main_h5_file[0])
    l1_ft, l1_dmt = getting_data(left_1[0])
    l2_ft, l2_dmt = getting_data(left_2[0])
    r1_ft, r1_dmt = getting_data(right_1[0])
    r2_ft, r2_dmt = getting_data(right_2[0])
    X = np.dstack([l2_ft, l1_ft, main_ft, r1_ft, r2_ft])
    Y = np.dstack([l2_dmt, l1_dmt, main_dmt, r1_dmt, r2_dmt])
    print(np.shape(X), np.shape(Y))
    fnout = main_h5_file[0].split(".h5")[0] + "_concatenated_beams.h5"
    with h5py.File(fnout, 'w') as f_f:
        f_f.attrs['cand_id'] = cand_id
        f_f.attrs['tcand'] = tcand
        f_f.attrs['dm'] = dm
        f_f.attrs['dm_opt'] = dm_opt
        f_f.attrs['snr'] = snr
        f_f.attrs['snr_opt'] = snr_opt
        f_f.attrs['width'] = width
        f_f.attrs['label'] = 0
        freq_time_dset = f_f.create_dataset('data_freq_time', data=X, compression="lzf")
        freq_time_dset.dims[0].label = b"time"
        freq_time_dset.dims[1].label = b"frequency"

        dm_time_dset = f_f.create_dataset('data_dm_time', data=Y, compression="lzf")
        dm_time_dset.dims[0].label = b"dm"
        dm_time_dset.dims[1].label = b"time"


if __name__ == '__main__':
    args = glob.glob(sys.argv[1])
    df = pd.DataFrame(args)
    df[1] = df[0].str.split("_tcand", expand = True)[0] + df[0].str.split("_dm", expand = True)[1].str.split("_beam", expand = True)[0] 
    gb = df.groupby([1])
    result = gb[0].unique()
    result = result.reset_index()
    for i, r in result.iterrows():
        main_h5_file = [s for s in r[0] if "main" in s] 
        beam_main = int(main_h5_file[0].split("_beam_")[1].split(".")[0]) 
        left_1 = [ s for s in r[0] if "_beam_" + str(beam_main -1) in s] 
        left_2 = [ s for s in r[0] if "_beam_" + str(beam_main -2) in s]  
        right_1 = [ s for s in r[0] if "_beam_" +  str(beam_main +1) in s] 
        right_2 = [ s for s in r[0] if "_beam_" + str(beam_main +2) in s] 
        concat(left_2, left_1, main_h5_file, right_1, right_2) 

