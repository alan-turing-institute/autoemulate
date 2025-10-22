# Spatiotemporal Emulation Framework

A unified YAML-based framework for training neural emulators on spatiotemporal data using The Well.

## Quick Start

```bash
# Run with example configs
python autoemulate/experimental/run_the_well_experiment.py \
  --config autoemulate/experimental/configs/turbulent_radiative_layer_2d.yaml

# Create a template config
python autoemulate/experimental/run_the_well_experiment.py --create-example
```

## Data Sources (Auto-detected)

The script automatically detects which data source to use:

### 1. The Well Native Datasets
```yaml
data:
  well_dataset_name: "turbulent_radiative_layer_2D"  # Triggers The Well native loading
  data_path: "../data/the_well/datasets"
  n_steps_input: 4
  n_steps_output: 1
  batch_size: 4
```

**Example:** `configs/turbulent_radiative_layer_2d.yaml`

### 2. File-Based Data
```yaml
data:
  data_path: "./data/bout"  # Path to HDF5/PyTorch files
  dataset_type: "bout"
  n_steps_input: 4
  n_steps_output: 10
  batch_size: 4
```

**Example:** `configs/bout.yaml`

### 3. Generated Data (from Simulators)
```yaml
simulator:
  type: "reaction_diffusion"  # or advection_diffusion
  parameters_range:
    feed_rate: [0.02, 0.06]
    kill_rate: [0.045, 0.065]
  n: 64
  T: 100.0
  dt: 1.0

data:
  n_train_samples: 200
  n_valid_samples: 20
  n_test_samples: 20
  n_steps_input: 4
  n_steps_output: 10
  batch_size: 4
```

**Examples:** `configs/reaction_diffusion_generated.yaml`, `configs/advection_diffusion_generated.yaml`

## Example Configurations

All configs are in `autoemulate/experimental/configs/`:

| Config | Data Source | Description |
|--------|-------------|-------------|
| `turbulent_radiative_layer_2d.yaml` | The Well Native | Turbulent flows with radiation |
| `bout.yaml` | File | BOUT++ plasma simulation data |
| `advection_diffusion_generated.yaml` | Generated | Transport phenomena (nu, mu parameters) |
| `reaction_diffusion_generated.yaml` | Generated | Gray-Scott pattern formation (F, k parameters) |

## Configuration Structure

```yaml
# Basic info
experiment_name: "my_experiment"
description: "What this experiment does"

# Emulator
emulator_type: "the_well_fno"  # FNO, AFNO, UNet variants
formatter_type: "default_channels_first"

# Model architecture
model_params:
  modes1: 16          # Fourier modes (x-direction)
  modes2: 16          # Fourier modes (y-direction)
  width: 32           # Hidden channels
  n_blocks: 4         # Number of FNO blocks

# Training
trainer:
  epochs: 100
  device: "mps"       # "cuda", "mps", or "cpu"
  optimizer_type: "adam"
  optimizer_params:
    lr: 0.001
  enable_amp: false   # Mixed precision (CUDA only)

# Teacher forcing (optional)
teacher_forcing:
  enabled: true
  schedule:
    - start_epoch: 0
      end_epoch: 50
      weight: 1.0
    - start_epoch: 50
      end_epoch: 100
      weight: 0.5

# Paths
paths:
  output_dir: "./outputs/my_experiment"
  model_save_path: "./outputs/my_experiment/artifacts/final_model.pt"

# Logging
log_level: "INFO"
verbose: true
```

## Emulator Types

- `the_well_fno` - Fourier Neural Operator (recommended)
- `the_well_afno` - Adaptive FNO
- `the_well_unet_classic` - Classic U-Net
- `the_well_unet_convnext` - ConvNext U-Net

## Device Support

### CPU
```yaml
trainer:
  device: "cpu"
```

### Apple Silicon (MPS)
```yaml
trainer:
  device: "mps"
```

### NVIDIA GPU (CUDA)
```yaml
trainer:
  device: "cuda"
  enable_amp: true      # Enable mixed precision
  amp_type: "float16"
```

## Output Structure

```
outputs/experiment_name/
├── config.yaml              # Configuration used
├── logs/
│   └── experiment_*.log    # Timestamped logs
├── checkpoints/
│   └── checkpoint_*.pt     # Training checkpoints
└── artifacts/
    └── final_model.pt      # Final trained model
```

## Common Tasks

### Quick Test Run
Set low epochs for testing:
```yaml
trainer:
  epochs: 10
```

### Adjust Memory Usage
Change batch size:
```yaml
data:
  batch_size: 8  # Increase if you have memory
```

### Change Dataset Size (Generated Data)
```yaml
data:
  n_train_samples: 500    # More data = better emulator
  n_valid_samples: 50
  n_test_samples: 50
```

## Adding New Datasets

> **Note:** Adding datasets requires editing source Python files (`config_models.py`, `run_the_well_experiment.py`, and your dataset class). This ensures type safety and explicit registration.

### For File-Based Data

If you have pre-existing HDF5 or PyTorch tensor files:

**1. Prepare your data in the expected format:**
- Shape: `[n_trajectories, n_timesteps, width, height, n_channels]`
- Save as HDF5 with key `"data"` or PyTorch `.pt` file

**2. Create a dataset class** (optional, if custom loading needed):

```python
# In autoemulate/experimental/data/spatiotemporal_dataset.py
class MyCustomDataset(AutoEmulateDataset):
    def read_data(self, data_path: str):
        """Load your custom data format."""
        # Load from your custom format
        data = load_my_custom_format(data_path)
        # Convert to expected shape
        self.data = torch.tensor(data, dtype=self.dtype)
```

**3. Register dataset type:**

Add to `DatasetType` enum in `config_models.py`:
```python
class DatasetType(str, Enum):
    MY_DATASET = "my_dataset"
```

**4. Register in script:**

Add to `get_dataset_class()` in `run_the_well_experiment.py`:
```python
def get_dataset_class(dataset_type: DatasetType):
    dataset_classes = {
        DatasetType.MY_DATASET: MyCustomDataset,
        # ... existing datasets
    }
```

Also import at the top of `run_the_well_experiment.py`:
```python
from autoemulate.experimental.data.spatiotemporal_dataset import (
    MyCustomDataset,
    # ... other imports
)
```

**5. Create config YAML:**
```yaml
data:
  data_path: "./data/my_dataset"
  dataset_type: "my_dataset"
  n_steps_input: 4
  n_steps_output: 10
  batch_size: 4
```

**Summary of files to edit:** `config_models.py` (enum), `run_the_well_experiment.py` (register + import), `spatiotemporal_dataset.py` (class)

### For Generated Data (Simulators)

> **Note:** Adding simulators requires editing source Python files (`config_models.py`, `run_the_well_experiment.py`, and creating your simulator class).

**1. Create simulator class:**

```python
# In autoemulate/simulations/my_simulator.py
from autoemulate.simulations.base import Simulator

class MySimulator(Simulator):
    def __init__(self, parameters_range, output_names, 
                 return_timeseries=False, n=64, T=10.0, dt=0.1):
        super().__init__(parameters_range, output_names)
        self.return_timeseries = return_timeseries
        self.n = n
        self.T = T
        self.dt = dt
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run simulation for parameters in x."""
        # Your simulation code here
        result = run_my_simulation(x, self.n, self.T, self.dt)
        return torch.tensor(result).reshape(1, -1)
    
    def forward_samples_spatiotemporal(self, n: int, 
                                       random_seed: int | None = None):
        """Generate n samples and reshape to spatiotemporal format."""
        # Generate samples
        X, y = self.forward_samples(n, random_seed)
        
        # Reshape to [n_samples, n_timesteps, width, height, n_channels]
        # ... your reshaping code
        
        return {
            "data": reshaped_data,
            "constant_scalars": None,
            "constant_fields": None,
        }
```

**2. Add to `SimulatorType` enum:**

In `config_models.py`:
```python
class SimulatorType(str, Enum):
    MY_SIMULATOR = "my_simulator"
```

**3. Register in `create_simulator()`:**

In `run_the_well_experiment.py`:
```python
def create_simulator(config: ExperimentConfig):
    sim_cfg = config.simulator
    
    if sim_cfg.type.value == "my_simulator":
        return MySimulator(
            parameters_range=sim_cfg.parameters_range,
            output_names=sim_cfg.output_names,
            return_timeseries=sim_cfg.return_timeseries,
            n=sim_cfg.n,
            T=sim_cfg.T,
            dt=sim_cfg.dt,
        )
```

Also import at the top of `run_the_well_experiment.py`:
```python
from autoemulate.simulations.my_simulator import MySimulator
```

**4. Create config YAML:**
```yaml
simulator:
  type: "my_simulator"
  parameters_range:
    param1: [min, max]
    param2: [min, max]
  n: 64
  T: 10.0
  dt: 0.1

data:
  n_train_samples: 200
  n_valid_samples: 20
  n_test_samples: 20
  n_steps_input: 4
  n_steps_output: 10
  batch_size: 4
```

**Summary of files to edit:** `config_models.py` (enum), `run_the_well_experiment.py` (register + import), `my_simulator.py` (class)

## Adding New Emulators

> **Note:** Adding emulators requires editing source Python files (`config_models.py`, `run_the_well_experiment.py`, and your emulator class). This ensures type safety and validation.

**1. Create emulator class:**

```python
# In autoemulate/experimental/emulators/the_well.py
from the_well.benchmark import models
from typing import ClassVar

class TheWellMyModel(TheWellEmulator):
    """My custom emulator using The Well framework."""
    
    # Specify the model class from The Well or your own
    model_cls: type[torch.nn.Module] = models.FNO  # or your custom model
    
    # Define default model parameters
    model_parameters: ClassVar[ModelParams] = {
        "modes1": 16,
        "modes2": 16,
        "width": 32,
        # ... other model-specific params
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

**For custom PyTorch models not in The Well:**

```python
class MyCustomModel(torch.nn.Module):
    """Your custom neural network architecture."""
    
    def __init__(self, in_channels, out_channels, width=32, **kwargs):
        super().__init__()
        # Define your architecture
        self.conv1 = torch.nn.Conv2d(in_channels, width, 3, padding=1)
        # ... more layers
    
    def forward(self, x):
        # Your forward pass
        return output

class TheWellMyCustomModel(TheWellEmulator):
    model_cls: type[torch.nn.Module] = MyCustomModel
    model_parameters: ClassVar[ModelParams] = {
        "width": 32,
    }
```

**2. Add to `EmulatorType` enum:**

In `config_models.py`:
```python
class EmulatorType(str, Enum):
    THE_WELL_MY_MODEL = "the_well_my_model"
```

**3. Register in `create_emulator()`:**

In `run_the_well_experiment.py`:
```python
def create_emulator(config: ExperimentConfig, datamodule):
    emulator_classes = {
        EmulatorType.THE_WELL_MY_MODEL: TheWellMyModel,
        # ... existing emulators
    }
```

**4. Import the emulator:**

In `run_the_well_experiment.py` at the top:
```python
from autoemulate.experimental.emulators.the_well import (
    TheWellMyModel,
    # ... other imports
)
```

**5. Create config YAML:**
```yaml
emulator_type: "the_well_my_model"
formatter_type: "default_channels_first"

model_params:
  modes1: 16
  modes2: 16
  width: 32
  # ... other model-specific parameters

trainer:
  epochs: 100
  device: "cuda"
  optimizer_type: "adam"
  optimizer_params:
    lr: 0.001
```

**Summary of files to edit:** `config_models.py` (enum), `run_the_well_experiment.py` (register + import), `the_well.py` (class)

**Tips:**
- Inherit from `TheWellEmulator` to get training loop, checkpointing, and logging for free
- Set `model_parameters` as class variable for defaults that can be overridden in config
- Use existing formatters (`default_channels_first`, `default_channels_first_with_time`) or create custom ones
- Available base models in The Well: `FNO`, `AFNO`, `UNet`, `ConvNextUNet`

## Troubleshooting

**CUDA out of memory:** Reduce `data.batch_size` or `model_params.width`

**Slow training:** Enable AMP on CUDA: `trainer.enable_amp: true`

**Data not found:** Check paths in config match your directory structure

## Design Philosophy

**Why explicit registration?**  
The framework uses explicit registration (editing Python files) rather than auto-discovery to ensure:
- ✅ Type safety through Pydantic validation
- ✅ Clear mapping of what's available
- ✅ IDE support for autocomplete and type checking
- ✅ No "magic" imports or hidden behavior

**Future improvement:** A decorator-based registry system could allow adding custom models without editing core files while maintaining most benefits.

## Further Information

See `configs/README.md` for detailed config documentation.
