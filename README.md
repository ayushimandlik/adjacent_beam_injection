# adjacent_beam_injection
In this project, the adjacent beam injection pipeline is described. It makes use of the software developed by Vivek Gupta. The complete software is described in: https://github.com/vg2691994/Furby.

The output of this pipeline is a catalog file names furbies.cat, and furby templates that can be read by the Furby_reader script.

adjacent_beams_injection_filterbank_version.py: This will output the filterbanks added to the furby (Mosk FRB) template.

adjacent_beams_getting_candidate_files_fetch.py: This will generate the injected Furbies in the adjacent and side beams, as well as output the candidate parameter file required to make candidates using FETCH candmaker described in https://github.com/devanshkv/fetch.


