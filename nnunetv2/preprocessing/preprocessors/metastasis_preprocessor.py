from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
import numpy as np

class MetastasisPreprocessor(DefaultPreprocessor):
    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager,
                     dataset_json):
        # Your custom label remapping
        # Example: keep class 1, convert class 2 to background
        seg[seg == 1] = 0  # Convert unwanted class to background
        seg[seg == 2] = 1
        
        # Continue with normal preprocessing
        return super().run_case_npy(data, seg, properties, plans_manager, 
                                   configuration_manager, dataset_json)