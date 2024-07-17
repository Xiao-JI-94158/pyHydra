"""
Python class that reads rawdata generated from Bruker ParaVision 360.3.5 environment

Testing examples are privided in the corresponing Jupyter Notebook (BrukerPV360.ipynb)
"""

# Official packages
import os
import copy
import re

from typing import List, Dict
from pprint import pprint
from collections import defaultdict

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline

#from pybaselines import Baseline

# In-house packages

POST_PROCESSING_PARAMETERS = {
    'is_verbose'                : False
}

RAW_DATA_FILE_LIST  = ['rawdata']
RAW_PARAM_FILE_LIST = ['acqp',  'method']

PROC_DATA_FILE_LIST = ['fid', '2dseq']
PROC_PARAM_FILE_LIST= ['reco', 'visu_pars']

DATA_COLLECTION_TEMPLATE = {
    'rawdata'   : None,
    '2dseq'     : None
}

PARAM_COLLECTION_TEMPLATE = {
    'ACQ'       : None,
    'PROC'      : None
}

class BrukerPV360Exp():
    """
        Basic Class that that read, stores, and (post-)processes data acquired from Bruker PavaVision 360 environment.
        
        Parameters:
        -----------

        exp_data_path: str, path of Bruker PV experiment
        
        Optional:
            'is_verbose'                : False                                        

    """
    
    def __init__(self, exp_dataset_path:str, **kwargs) -> None:
        """
        0. update params for post-processing:
    
        1. validate dataset files:
        
        2. update param_collection:

        3. update data_collection:

        """
        # Step 0
        self.post_processing_params = self._update_post_processing_params(kwargs)
        
        if (self.post_processing_params['is_verbose']):
            print(f'Input directory: {exp_dataset_path}')
        
        # Step 1
        self.dataset = self._validate_dataset_files(exp_dataset_path)

        # Step 2
        self._update_acq_params()  
        
        # Step 3
        self._update_dataset_data()

        
    def _update_post_processing_params(self, kwargs):
        """
        parse possible tags for post-processing
        """
        recon_params = copy.deepcopy(POST_PROCESSING_PARAMETERS)
        recon_params.update((k, kwargs[k]) for k in (recon_params.keys() & kwargs.keys()) )
        return recon_params  

    def _validate_dataset_files(self, exp_dataset_path)->Dict:
        """
        Confirm that the given path of experimental dataset is valid
        """

        raw_data_paths = self._extract_raw_path( exp_dataset_path, key_list=RAW_DATA_FILE_LIST)
        raw_param_paths = self._extract_raw_path( exp_dataset_path, key_list=RAW_PARAM_FILE_LIST)

        processed_data_dir = os.path.join(exp_dataset_path, 'pdata')
        
        proc_data_paths=self._extract_proc_path( processed_data_dir, key_list=PROC_DATA_FILE_LIST)                                    
        proc_param_paths=self._extract_proc_path( processed_data_dir, key_list=PROC_PARAM_FILE_LIST)
 
        if (self.post_processing_params['is_verbose']):
            print('end of dataset file validation')
            pprint(f'Found raw data files:')
            pprint(raw_data_paths)
            pprint(f'Found raw parameter files:')
            pprint(raw_param_paths)
            pprint(f'Found proc data files: ')
            pprint(proc_data_paths)
            pprint(f'Found proc parameter files: ')
            pprint(proc_param_paths)

        dataset = {
                    'ACQ_DATA'  : raw_data_paths ,
                    'PROC_DATA' : proc_data_paths,
                    'ACQ_PARAM' : raw_param_paths,
                    'PROC_PARAM': proc_param_paths,
                   }

        return dataset
        

    def _extract_raw_path(self, raw_file_dir, key_list):
        raw_paths = defaultdict(list)

        for rfile_key in key_list:
            for rfile_name in os.scandir(raw_file_dir):
                rfile_path = os.path.join(rfile_name)
                if (rfile_key in rfile_path):
                    if (os.path.isfile(rfile_path)):
                        raw_paths[rfile_key].append(rfile_path)
        return dict(raw_paths)

    def _extract_proc_path(self, proc_file_dir, key_list):
        proc_paths = defaultdict(list)

        if (os.path.isdir(proc_file_dir)):
            for pfile_key in key_list:
                for proc_nbr in os.scandir(proc_file_dir):
                    proc_nbr_path = os.path.join(proc_nbr)
                    if (os.path.isdir(proc_nbr_path)):
                        for pfile_name in os.scandir(proc_nbr_path):
                            pfile_path = os.path.join(pfile_name)
                            if ((pfile_key in pfile_path)):
                                if (os.path.isfile(pfile_path)):
                                    proc_paths[pfile_key].append(pfile_path)
        return dict(proc_paths)

    def _update_acq_params(self):
        """
        """
        for key, value in self.dataset['ACQ_PARAM'].items():
            temp_dict = self._read_param_dicts(value[0])

        self.dataset['ACQ_PARAM'][key] = temp_dict

    def _read_param_dicts(self, param_file_path):
        """
        Read a Bruker MRI experiment's parameter files to a dictionary.

        Ref: https://github.com/jdoepfert/brukerMRI
        """

        param_dict = {}

        with open(param_file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # when line contains parameter
                if line.startswith('##$'):

                    (param_name, current_line) = line[3:].split('=') # split at "="
                    
                    # if current entry (current_line) is arraysize
                    if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                        value = self._parse_array(f, current_line)

                    # if current entry (current_line) is struct/list
                    elif current_line[0] == "(" and current_line[-3:-1] != " )":

                        # if neccessary read in multiple lines
                        while current_line[-2] != ")":
                            current_line = current_line[0:-1] + f.readline()

                        # parse the values to a list
                        value = [self._parse_single_value(x) for x in current_line[1:-2].split(', ')]

                    # otherwise current entry must be single string or number
                    else:
                        value = self._parse_single_value(current_line)

                    # save parsed value to dict
                    param_dict[param_name] = value

        return param_dict

    def _flatten(self, l):
        if not isinstance(l, list):
            return [l]
        flat = []
        for sublist in l:
            flat.extend(self._flatten(sublist))
        return flat

    def _parse_array(self, current_file, line):
        """
        Ref: https://github.com/jdoepfert/brukerMRI
        """
        # extract the arraysize and convert it to numpy
        line = line[1:-2].replace(" ", "").split(",")
        try:
            arraysize = np.array([int(x) for x in line])
        except ValueError:
            return line

        # then extract the next line
        vallist = current_file.readline().split()

        # if the line was a string, then return it directly
        try:
            float(vallist[0])
        except ValueError:
            return " ".join(vallist)

        # include potentially multiple lines
        while len(vallist) != np.prod(arraysize):
            new_line =  current_file.readline()
            if ('##$' in new_line):
                break
            elif ('@' in new_line):
                if ('*' in new_line):
                    vallist = vallist + new_line.split()
                else:
                    break
            else:
                vallist = vallist + new_line.split()

        # try converting to int, if error, then to float
        for idx, val in enumerate(vallist):
            try:
                vallist[idx] = int(val)
            except ValueError:
                try:
                    vallist[idx] = float(val)
                except ValueError:
                        if (('@' in val) and ('*' in val)):
                            res = re.findall(r"\d+", val)
                            reps, value = [int(x) for x in res]
                            vallist[idx] = [value] * reps
                            vallist= self._flatten(vallist)

        """
        # This block below is the original code from Ref: https://github.com/jdoepfert/brukerMRI
        # For our purpose, we return all numerical types in format of numpy.ndarray, regardless of its length

            # convert to numpy array
            if len(vallist) > 1:
                return np.reshape(np.array(vallist), arraysize)
            # or to plain number
            else:
                return vallist[0]
        """
        return np.reshape(np.array(vallist), arraysize)

    def _parse_single_value(self, val):
        """
        Ref: https://github.com/jdoepfert/brukerMRI
        """
        try: # check if int
            result = int(val)
        except ValueError:
            try: # then check if float
                result = float(val)
            except ValueError:
                # if not, should  be string. Remove  newline character.
                result = val.rstrip('\n')

        return result    
    
    def _update_dataset_data(self):
        for key, value in self.dataset['ACQ_DATA'].items():
            for idx, rd_path in enumerate(value):
                self.dataset['ACQ_DATA'][key][idx] = self._process_rawdata(rd_path)

        for key, value in self.dataset['PROC_DATA'].items():
            for idx, rd_path in enumerate(value):
                if (self.post_processing_params['is_verbose']):
                    print(key, value, idx)
                if key=='2dseq':
                    self.dataset['PROC_DATA'][key][idx] = self._process_2dseq(rd_path)
                elif key=='fid':
                    pass
                else:
                    pass
                    


    
    def _process_rawdata(self, rawdata_path):
        """
        Read binary fid into cmplx128 format, and partition into transients.
        """
        #_raw_fid_dtype = self.dataset['ACQ_PARAM']['ACQ_word_size']
        raw_fid = np.fromfile(file=rawdata_path, dtype='int32')

        cmplx_fid = np.asarray(raw_fid[0::2, ...] + 1j * raw_fid[1::2, ...])
        cmplx_fid.astype(np.complex128)

        
        return cmplx_fid

    
    def _process_2dseq(self, rawdata_path):
        """
        Read and reshape the 2dseq image, which is reconstructed with Bruker algorithm and stored in Bruker format.
        """
        raw_2dseq = np.fromfile(file=rawdata_path, dtype='int16')
        
        return raw_2dseq