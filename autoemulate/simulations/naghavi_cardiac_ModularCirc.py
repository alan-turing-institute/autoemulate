from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from ModularCirc.Models.NaghaviModel import NaghaviModel, NaghaviModelParameters
from ModularCirc.Solver import Solver
from tqdm.notebook import tqdm  # For Jupyter notebook progress bar
from autoemulate.simulations import circ_utils

import json
import numpy as np

def extract_parameter_ranges(json_file_path):
    """
    Extract parameter ranges from a JSON file and return them in the format:
    {
        "param1": (min_val, max_val),  # MUST be tuple of exactly two floats
        "param2": (min_val, max_val),
        ...
    }
    """

    with open(json_file_path) as file:
        dict_parameters = json.load(file)
        circ_utils.condense_dict_parameters(dict_parameters)
        circ_utils.dict_parameters_condensed_range
    
    return circ_utils.dict_parameters_condensed_range

class NaghaviSimulator:
    def __init__(self,
                 n_cycles: int = 40, dt: float = 0.001):
        """Initialize simulator with parameter ranges"""
        self.n_cycles = n_cycles
        self.dt = dt
        self.time_setup = {
            "name": "HistoryMatching",
            "ncycles": n_cycles,
            "tcycle": 1.0,
            "dt": dt,
            "export_min": 1
        }

    def run_simulation(self, params: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Run simulation and return all available time series outputs"""
        # Set parameters
        parobj = NaghaviModelParameters()
        for param_name, value in params.items():
            if param_name == "T":
                continue
            try:
                obj, param = param_name.split('.')
                parobj._set_comp(obj, [obj], **{param: value})
            except Exception as e:
                print(f"Error setting parameter {param_name}: {e}")
                return None
        
        # Set cycle time
        t_cycle = params.get("T", 1.0)
        self.time_setup["tcycle"] = t_cycle
        
        # Run simulation
        try:
            model = NaghaviModel(time_setup_dict=self.time_setup, parobj=parobj, suppress_printing=True)
            solver = Solver(model=model)
            solver.setup(suppress_output=True, optimize_secondary_sv=False, method='LSODA')
            solver.solve()
            
            if not solver.converged:
                print("Solver did not converge.")
                return None
        except Exception as e:
            print(f"Simulation error: {e}")
            return None
            
        # Collect outputs - only time series data
        raw_results = {}
        output_names = []
        
        for component_name, component_obj in model.components.items():
            for attr_name in dir(component_obj):
                # Skip special methods and kwargs
                if (not attr_name.startswith('_') and 
                    not attr_name == 'kwargs' and
                    not callable(getattr(component_obj, attr_name))):
                    
                    try:
                        attr = getattr(component_obj, attr_name)
                        if hasattr(attr, 'values'):
                            full_name = f"{component_name}.{attr_name}"
                            raw_results[full_name] = attr.values.tolist()
                            output_names.append(full_name)
                    except Exception:
                        continue
        
        # Store output names after first successful simulation
        if not hasattr(self, '_output_names'):
            self._output_names = output_names
        
        return raw_results

    def run_batch_simulations(self, samples: Union[List[Dict[str, float]], pd.DataFrame]) -> List[Dict[str, float]]:
        """Run batch simulations with progress tracking"""
        if isinstance(samples, pd.DataFrame):
            samples = samples.to_dict(orient='records')
            
        results = []
        successful = 0
        
        for sample in tqdm(samples, desc="Running simulations", unit="sample"):
            # Run simulation for each sample
            result = self.run_simulation(sample)
            if result is not None:
                results.append({**sample, **result})
                successful += 1
        
        print(f"Successfully completed {successful}/{len(samples)} simulations ({successful/len(samples)*100:.1f}%)")
        return results