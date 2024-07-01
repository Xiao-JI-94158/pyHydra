"""
Python class that reads rawdata generated from Bruker ParaVision 6 environment

Testing examples are privided in the corresponing Jupyter Notebook (BrukerPV6.ipynb)
"""

# Official packages
import os
import copy

from typing import List, Dict
from pprint import pprint

# Third-party packages
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline, CubicSpline

from pybaselines import Baseline

# In-house packages

POST_PROCESSING_PARAMETERS = {
    'is_verbose'                : False
}

RAW_DATA_SET = {
    'rawdata'   : 'rawdata',
    'fid'       : 'pdata/1/fid_proc.64',
    # pdata subdir might not exist due to user config (not performing factory reconstruction)
    '2dseq'     : 'pdata/1/2dseq',
    'dicom'     : 'pdata/1/dicom',
}

ACQ_PARAM_SET = {
    'acqp'      : 'acqp',
    'acqp.out'  : 'acqp.out',
    'configscan': 'configscan'
    'method'    : 'method'
}

RECO_PARAM_SET = {
    'id'        : 'pdata/1/id',
    'methreco'  : 'pdata/1/methreco',
    'reco'      : 'pdata/1/reco',
    'reco.out'  : 'pdata/1/reco.out',
    'visu_pars' : 'pdata/1/visu_pars'
}


DATA_COLLECTION_TEMPLATE = {
    'rawdata'   : None,
    '2dseq'     : None,
    'dicom'     : None
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
    
        1. validate experiment dataset:
            1.1 data files:
                must        : rawdata.jobX, 
                optional    : fid_proc.64, 2dseq, dicom
            1.2 param files:

        
        2. update dataset_dict['PARAM']:

        3. update data_collection:

        4. perform reconstruction:

        """
    
        self.post_processing_params = self._update_post_processing_params(kwargs)

        if (self.post_processing_params['is_verbose']):
            print(exp_dataset_path)
        
        self.dataset = {"DATA": None, "PARAM": None}

        self._validate_dataset_files(exp_dataset_path)

        self._update_dataset_param()       
        self._update_dataset_data()
        
        self.dataset['DATA']['2dseq'] = self._process_2dseq()
        self.dataset['DATA']['fid'] = self._process_fid()    
        
    def _update_post_processing_params(self, kwargs):
        """
        parse possible tags for post-processing
        """
        recon_params = copy.deepcopy(POST_PROCESSING_PARAMETERS)
        recon_params.update((k, kwargs[k]) for k in (recon_params.keys() & kwargs.keys()) )
        return recon_params  

    def _validate_dataset_files(self, exp_dataset_path):
        """
        Confirm that the given path of experimental dataset is valid
        """
        if (not (os.path.isdir(exp_dataset_path))):
            raise OSError(f"Given directory of Experiment ({exp_dataset_path}) does not exist")
        
        self._validate_data_files(exp_dataset_path)
        self._validate_param_files(exp_dataset_path)

        

    def _validate_data_files(self, exp_dataset_path)->Dict:
        """
        1.1 data files:
            must        : rawdata.jobX
            optional    : 2dseq
                          fid_proc.64
                          dicom
        """
        data_dict = RAW_DATA_SET 
     
        self.dataset['DATA'] = self._complete_abs_path(data_dict, exp_dataset_path) 

        for key, val in self.dataset['DATA'].items():
            if (not os.path.exists(val)):
                
                self.dataset['DATA'][key] = None
       
        if ((self.dataset['DATA']['fid'] == None) and (self.dataset['DATA']['ser'] == None)):
                raise FileNotFoundError(f"Cannot find raw data file, neither fid nor ser, in the given directory of Experiment ({exp_dataset_path})")
        
        if (self.post_processing_params['is_verbose']):
            print('end of _validate_data_files')
            pprint(self.dataset['DATA'])

    def _validate_param_files(self, exp_dataset_path)->Dict:
        """
        1.2 param files:
            must        : acqp, method, visu_pars
            optional    : acqu, acqus, procs, reco        
        """
        param_dict = RAW_PARAM_SET
        self.dataset['PARAM'] = self._complete_abs_path(param_dict, exp_dataset_path)

        for key, val in self.dataset['PARAM'].items():
            if (not os.path.isfile(val)):
                self.dataset['DATA'][key] = None

        for key in ['acqp', 'method', 'visu_pars']:
            if (self.dataset['PARAM'][key] == "None"):
                raise FileNotFoundError(f"Cannot find {key} file in the given directory of Experiment ({exp_dataset_path})")
        


    def _complete_abs_path(self, dp_dict, exp_dataset_path):
        """
        """
        ret_dict = copy.deepcopy(dp_dict)
        
        for key, value in ret_dict.items():
            abs_path = os.path.join(exp_dataset_path, value)
            ret_dict[key] = abs_path
        
        return ret_dict


    def _update_dataset_param(self):
        """
        """
        param_dict = {}
        for key, value in self.dataset['PARAM'].items():
            temp_dict = self._read_param_dicts(value)
            param_dict = (param_dict | temp_dict)
        
        self.dataset['PARAM'] = param_dict
        
    
    def _update_dataset_data(self):

        data = copy.deepcopy(DATA_COLLECTION_TEMPLATE)
        
        if self.dataset['DATA']['2dseq']:
            data['2dseq'] = self.dataset['DATA']['2dseq']
        if self.dataset['DATA']['fid']:
            data['fid'] = self.dataset['DATA']['fid']
        if self.dataset['DATA']['ser']:
            data['ser'] = self.dataset['DATA']['ser']

        self.dataset['DATA'] = data

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
        

    def _parse_array(self, current_file, line):
        """
        Ref: https://github.com/jdoepfert/brukerMRI
        """
        # extract the arraysize and convert it to numpy
        line = line[1:-2].replace(" ", "").split(",")
        arraysize = np.array([int(x) for x in line])

        # then extract the next line
        vallist = current_file.readline().split()

        # if the line was a string, then return it directly
        try:
            float(vallist[0])
        except ValueError:
            return " ".join(vallist)

        # include potentially multiple lines
        while len(vallist) != np.prod(arraysize):
            vallist = vallist + current_file.readline().split()

        # try converting to int, if error, then to float
        try:
            vallist = [int(x) for x in vallist]
        except ValueError:
            vallist = [float(x) for x in vallist]

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

    def _process_2dseq(self):
        """
        Read and reshape the 2dseq image, which is reconstructed with Bruker algorithm and stored in Bruker format.
        """
        _raw_2dseq_dtype = self.dataset['PARAM']['VisuCoreWordType']
        _raw_2dseq_b_order = self.dataset['PARAM']['VisuCoreByteOrder']

        if ((_raw_2dseq_dtype == '_16BIT_SGN_INT') and (_raw_2dseq_b_order == 'littleEndian')):
            raw_2dseq = np.fromfile(file=self.dataset['DATA']['2dseq'], dtype='int16')
        
        data_shape = np.append(self.dataset['PARAM']['NR'], -1)

        raw_2dseq = np.reshape(raw_2dseq, data_shape)
        
        return raw_2dseq
    
    def _process_fid(self):
        """
        Read binary fid into cmplx128 format, and partition into transients.
        """
        raw_fids = self._read_binary_fid()
        raw_fids = self._deserialize_binary_fid(raw_fids)
        raw_fids = np.asarray(np.array_split(raw_fids, self.dataset['PARAM']["NR"]))
        return raw_fids

    def _read_binary_fid(self) -> np.ndarray:
        """
        """
        _raw_fid_dtype = self.dataset['PARAM']['GO_raw_data_format']
        if (_raw_fid_dtype == 'GO_32BIT_SGN_INT') :
            fid = np.fromfile(file=self.dataset['DATA']['fid'], dtype='int32')

        else:
            raise TypeError( f'Raw FID data in Unknown Datatype ({_raw_fid_dtype})' )
        return fid
    
    def _deserialize_binary_fid(self, fid) -> np.ndarray:
        fid = np.asarray(fid[0::2, ...] + 1j * fid[1::2, ...])
        fid.astype(np.complex128)
        return fid
    
    def _process_raw(self, raw_type):
        raw_fids = self._read_binary(raw_type)
        raw_fids = self._deserialize_binary_cmplx(raw_fids)
        raw_fids = np.asarray(np.array_split(raw_fids, self.dataset['PARAM']["NR"]))
        return raw_fids

    def _read_binary(self, raw_type) -> np.ndarray:
        """
        """
        _raw_fid_dtype = self.dataset['PARAM']['GO_raw_data_format']
        if (_raw_fid_dtype == 'GO_32BIT_SGN_INT') :
            fid = np.fromfile(file=self.dataset['DATA'][raw_type], dtype='int32')

        else:
            raise TypeError( f'Raw FID data in Unknown Datatype ({_raw_fid_dtype})' )
        return fid
    
    def _deserialize_binary_cmplx(self, fid) -> np.ndarray:
        fid = np.asarray(fid[0::2, ...] + 1j * fid[1::2, ...])
        fid.astype(np.complex128)
        return fid
    
    def _fit_proj_baseline(self, proj, lambda_fit):
        baseline_fitter = Baseline(x_data=proj)                     
        return baseline_fitter.aspls(proj, lam=lambda_fit)[0]

    def _normalize_splines(self):
        
        
        
        return NotImplemented