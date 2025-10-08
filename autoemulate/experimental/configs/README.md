# Example Configurations

Example YAML configurations for different data sources and emulation scenarios.

## Available Configs

### `turbulent_radiative_layer_2d.yaml`
- **Data:** The Well native dataset
- **Use:** Turbulent flows with radiation
- **Device:** MPS (Apple Silicon)
- **Samples:** 6,984 train / 873 validation / 873 test

```bash
python autoemulate/experimental/run_the_well_experiment.py \
  --config autoemulate/experimental/configs/turbulent_radiative_layer_2d.yaml
```

### `bout.yaml`
- **Data:** File-based (BOUT++ plasma simulations)
- **Use:** Plasma physics, fusion research
- **Device:** MPS
- **Location:** `./autoemulate/experimental/exploratory/data/bout/`

```bash
python autoemulate/experimental/run_the_well_experiment.py \
  --config autoemulate/experimental/configs/bout.yaml
```

### `advection_diffusion_generated.yaml`
- **Data:** Generated from simulator
- **Use:** Transport phenomena, fluid dynamics
- **Device:** MPS
- **Parameters:** nu (viscosity) [0.0001, 0.01], mu (advection) [0.5, 2.0]
- **Samples:** 200 train / 20 validation / 20 test

```bash
python autoemulate/experimental/run_the_well_experiment.py \
  --config autoemulate/experimental/configs/advection_diffusion_generated.yaml
```

### `reaction_diffusion_generated.yaml`
- **Data:** Generated from simulator
- **Use:** Pattern formation, chemical dynamics (Gray-Scott model)
- **Device:** MPS
- **Parameters:** feed_rate [0.02, 0.06], kill_rate [0.045, 0.065]
- **Samples:** 200 train / 20 validation / 20 test

```bash
python autoemulate/experimental/run_the_well_experiment.py \
  --config autoemulate/experimental/configs/reaction_diffusion_generated.yaml
```

## Quick Customizations

### Change Device
```yaml
trainer:
  device: "cuda"  # or "mps" or "cpu"
```

### Enable Mixed Precision (CUDA only)
```yaml
trainer:
  enable_amp: true
  amp_type: "float16"
```

### Quick Test (Fewer Epochs)
```yaml
trainer:
  epochs: 10
```

### Adjust Memory Usage
```yaml
data:
  batch_size: 8  # Increase if you have GPU memory
```

## Data Source Detection

The script auto-detects the data source:

1. **The Well Native** - If `data.well_dataset_name` is set
2. **File-Based** - If `data.data_path` is set (without `well_dataset_name`)
3. **Generated** - If `simulator` section exists

## Creating Your Own Config

Copy an existing config and modify:

1. Change `experiment_name` and `description`
2. Adjust data source (see parent README.md)
3. Tune model parameters (`modes1`, `modes2`, `width`)
4. Set training parameters (`epochs`, `device`, `lr`)
5. Run!

See `../README.md` for full configuration documentation.
