# adjacent_beam_injection
In this project, the adjacent beam injection pipeline is described. It makes use of the software developed by Vivek Gupta. The complete software is described in: https://github.com/vg2691994/Furby.

The output of this pipeline is a catalog file names furbies.cat, and furby templates that can be read by the Furby_reader script.

adjacent_beams_injection_filterbank_version.py: This will output the filterbanks added to the furby (Mosk FRB) template.

adjacent_beams_getting_candidate_files_fetch.py: This will generate the injected Furbies in the adjacent and side beams, as well as output the candidate parameter file required to make candidates using FETCH candmaker described in https://github.com/devanshkv/fetch.

furby_beam_info.csv: This file is the output from the adjacent_beams_getting_candidate_files_fetch.py file. It gives the "DM_inj,SNR_inj,beam,furby_id,main_beam,side_beam,tstamp" values for convenience.

fetch_path_to_make_cands.csv: Example of the fetch candidate parameter file.

concatenation_of_fetch_candidate_files.py: This script can be used in the concatenation of the h5 files that have been injected with the furby candidate. The output of the files must indicate the beam numbers and whether the file is the main beam or side beam. This can be actieved by making small changes to the candmaker and candidate script in the FETCH candidate maker. The adjacent_beams_getting_candidate_files_fetch.py script outputs this information onto the parameter file, which can be used to modify the name of the candidates.

adjecent_beams_removal.py: This is the CNN code. The weights will be stored as "weights.best.5D_FINAL_learning_rate_0.001_batch_150_epochs_100.hdf5", and the model is stored as a JSON file: model.json. The "training.csv file is of the format:
concatenated_beam_filename.h5, label

CNN_neural_net_get_AOC.py: This file gives the predictions and the value for area under curve. The predictions are stored in a file called predictions_output.py
