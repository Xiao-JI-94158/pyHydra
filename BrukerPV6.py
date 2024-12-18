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
    'is_verbose'                : False, 
    'does_update_pdata'          : True
}



class BrukerPV6Exp():
    """
        Graneral Basic Class that that read, stores, and (post-)processes data acquired from Bruker PavaVision 6 environment.
        
        Parameters:
        -----------

        exp_data_path: str, path of Bruker PV experiment
        
        Optional:
            'is_verbose'                : False                                        

    """
    
    def __init__(self, exp_dataset_path:str, **kwargs) -> None:
        """
        Update20241213:
        We are going to rewrite the logic here:
        0. update parameters for post-processing

        1. look for essential k-space data and parameters for data acquisition, then update them
            1.1 find raw readout data: FID or SER, at least one should be present
            1.2 find parameter files for data acquisition: acqp and method, both must be there
            1.3 update k-space data
            1.4 update acq params
        
        2. look for Bruker factory recon
            2.1 verify the existence of Bruker factory recon data:
                2.1.1 if None found, return None as result
                2.1.2 if found, return the number of sets of Bruker factory recon data
            2.2 find Bruker factory recon data: 2dseq
            2.3 find Bruker factory recon parameter files: visu_pars, procs
            2.4 update Bruker factory recon data
            2.5 update Bruker factory recon params

        3. (optional) in-house recon

        """
    
        self.post_processing_params = self._update_post_processing_params(kwargs)

        if (self.post_processing_params['is_verbose']):
            print(exp_dataset_path)
        self.exp_dataset_path = exp_dataset_path
        self.dataset = {"DATA": {}, "PARAM": {}}

        self._process_rawdata_files()

        if (self.post_processing_params['does_update_pdata']):
            self._process_pdata_files()
        else:
            self._ignore_pdata_files()

    def _update_post_processing_params(self, kwargs):
        """
        parse possible tags for post-processing
        """
        recon_params = copy.deepcopy(POST_PROCESSING_PARAMETERS)
        recon_params.update((k, kwargs[k]) for k in (recon_params.keys() & kwargs.keys()) )
        return recon_params  

    def _process_rawdata_files(self):
        """
        1. _validate_acq_params
        2. _validate_rawdata_binary
        """

        self._validate_acq_params()
        self._validate_rawdata_binary()

        pass


    def _validate_acq_params(self):
        acqp_path = os.path.join(self.exp_dataset_path, 'acqp')
        if (os.path.exists(acqp_path)):
            self.dataset['PARAM'].update({'acqp': self._read_param_dicts(acqp_path)})
        else:
            self.dataset['PARAM']['acqp'] = None
            raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does not contain acqp file")
        
        method_path = os.path.join(self.exp_dataset_path, 'method')
        if (os.path.exists(method_path)):
            self.dataset['PARAM']['method'] = self._read_param_dicts(method_path)
        else:
            self.dataset['PARAM']['method'] = None
            raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does not contain method file")
        
        visu_pars_path = os.path.join(self.exp_dataset_path, 'visu_pars')
        if (os.path.exists(method_path)):
            self.dataset['PARAM']['visu_pars'] = self._read_param_dicts(visu_pars_path)
        else:
            self.dataset['PARAM']['visu_pars'] = None
            raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does not contain visu_pars file")
        


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

    def _validate_rawdata_binary(self):
                
        #   Confirm that the given path of experimental dataset is valid
        if (os.path.isdir(self.exp_dataset_path)):
            fid_path = os.path.join(self.exp_dataset_path, 'fid')
            if (os.path.exists(fid_path)):
                self.dataset['DATA']['fid_path'] = fid_path
            else:
                ser_path = os.path.join(self.exp_dataset_path, 'ser')
                if (os.path.exists(ser_path)):
                    self.dataset['DATA']['fid_path'] = ser_path
                else:
                    self.dataset['DATA']['fid_path'] = None
                    raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does not contain any binary data")

        else:
            raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does not exist")
        if (self.dataset['DATA']['fid_path']):
            self.dataset['DATA']['fid'] = self._process_fid()  
  
    def _process_fid(self):
        """
        Read binary fid into cmplx128 format, and partition into transients.
        """
        raw_fids = self._read_binary_fid()
        raw_fids = self._deserialize_binary_fid(raw_fids)
        #   Pending array NR-reshaping 
        #   raw_fids = np.asarray(np.array_split(raw_fids, self.dataset['PARAM']["NR"]))
        return raw_fids

    def _read_binary_fid(self) -> np.ndarray:
        """
        """
        _raw_fid_dtype = self.dataset['PARAM']['acqp']['GO_raw_data_format']
        if (_raw_fid_dtype == 'GO_32BIT_SGN_INT') :
            fid_dtype = 'int32'
        elif (_raw_fid_dtype == 'GO_16BIT_SGN_INT'):
            fid_dtype = 'int16'
        else:
            raise TypeError( f'Raw FID data in Unknown Datatype ({_raw_fid_dtype})' )
        
        fid = np.fromfile(file=self.dataset['DATA']['fid_path'], dtype=fid_dtype)
        return fid
    
    def _deserialize_binary_fid(self, fid) -> np.ndarray:
        fid = np.asarray(fid[0::2, ...] + 1j * fid[1::2, ...])
        fid.astype(np.complex128)
        return fid
 
    def _ignore_pdata_files(self):
        self.dataset['DATA']['2dseq'] = None
        self.dataset['PARAM']['procs'] = None
        self.dataset['PARAM']['reco'] = None
        self.dataset['PARAM']['p_visu_pars'] = None

    def _process_pdata_files(self):
        """
        1. screen pdata folder and get number of sub-directories
            1.1 os.walk the pdata dir
            1.2 generate list of 2dseq paths
            1.3 generate list of procs paths
            1.4 generate list of reco paths
            1.5 generate list of p_visu_pars paths
        2. update pdata 2dseq and respective reco_params
        """



        pdata_dir_path = os.path.join(self.exp_dataset_path, 'pdata')
        # Exist pdata dir
        if (os.path.exists(pdata_dir_path)):
            pdata_walk_through_list = list(os.walk(pdata_dir_path))

            # pdata dir has content
            pdata_dir_content = pdata_walk_through_list[0][1]
            if ( pdata_dir_content ):
                for idx, sub_dir_info in enumerate(pdata_walk_through_list[1:]):
                    self.dataset['DATA'][pdata_dir_content[idx]] = {}
                    self.dataset['PARAM'][pdata_dir_content[idx]] = {}
                    sub_dir_path = sub_dir_info[0]


                    if ('procs' in sub_dir_info[2]):
                        path_procs = os.path.join(sub_dir_path, 'procs')
                        self.dataset['PARAM'][pdata_dir_content[idx]]['procs'] = self._read_param_dicts(path_procs)
                    
                    if ('reco' in sub_dir_info[2]):
                        path_reco = os.path.join(sub_dir_path, 'reco')
                        self.dataset['PARAM'][pdata_dir_content[idx]]['reco'] = self._read_param_dicts(path_reco)

                    if ('visu_pars' in sub_dir_info[2]):
                        path_p_visu_pars = os.path.join(sub_dir_path, 'visu_pars')
                        self.dataset['PARAM'][pdata_dir_content[idx]]['visu_pars'] = self._read_param_dicts(path_p_visu_pars)                                            

                    
                    if ('2dseq' in sub_dir_info[2]):
                        path_2dseq = os.path.join(sub_dir_path, '2dseq')
                        self.dataset['DATA'][pdata_dir_content[idx]]['2dseq'] = self._process_2dseq(pdata_dir_content[idx], path_2dseq)

                pass
            # empty pdata dir
            else:
                self._ignore_pdata_files()
                raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) contain empty pdata folder.")
            
        # No pdata dir
        else:
            self._ignore_pdata_files()
            raise OSError(f"Given directory of Experiment ({self.exp_dataset_path}) does contain pdata folder.")
        


    def _process_2dseq(self, pdata_idx, path_2dseq):
        """
        Read and reshape the 2dseq image, which is reconstructed with Bruker algorithm and stored in Bruker format.
        """
        _raw_2dseq_dtype = self.dataset['PARAM'][pdata_idx]['visu_pars']['VisuCoreWordType']
        _raw_2dseq_b_order = self.dataset['PARAM'][pdata_idx]['visu_pars']['VisuCoreByteOrder']

        if (_raw_2dseq_dtype == '_32BIT_SGN_INT'):
            _raw_2dseq_dtype = 'int32'
        elif (_raw_2dseq_dtype == '_16BIT_SGN_INT'):
            _raw_2dseq_dtype = 'int16'
        elif (_2dseq_dtype=='_32BIT_FLOAT'):
            _2dseq_dtype = 'float32'
        else:
            raise ValueError(f"Unknown VisuCoreWorkType ({_raw_2dseq_dtype}). Need to update code to accomodate.")


        if (_raw_2dseq_b_order == 'littleEndian'):
            return np.fromfile(file=path_2dseq, dtype=_raw_2dseq_dtype)
        else:        
            return None 

    def _fit_proj_baseline(self, proj, lambda_fit):
        baseline_fitter = Baseline(x_data=proj)                     
        return baseline_fitter.aspls(proj, lam=lambda_fit)[0]

    def _normalize_splines(self):
        return NotImplemented
        
    