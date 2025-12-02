# Data Synthesis Module

The data synthesis module converts EDA design data into learning-ready formats for training large models on chip routing pattern generation.

## Overview

This module processes physical design files and generates structured datasets suitable for transformer-based sequence learning. It handles the complex task of extracting routing patterns, spatial locations, and connectivity graphs from chip design data.

## Data Types Generated

- `net_seqs`: Network sequence representations with driver/load information
- `pin2pin_pattern_seqs`: Pin-to-pin routing pattern sequences
- `pin2pin_loc_seqs`: Spatial location sequences for routing paths  
- `design_graph`: Design-level connectivity and spatial overlap graphs

## Usage

### Individual Design Processing
```bash
python -m data_synthesis.main_aggregation --design_list nvdla shanghai_MS --output_dir /path/to/output
```

### Batch Processing Multiple Designs
```bash
python -m data_synthesis.main_aggregation --design_list design1 design2 design3 --data_types net_seqs pin2pin_pattern_seqs
```

### Configuration Options
```bash
# Enable detailed logging
python -m data_synthesis.main_aggregation --enable_dataset_logs

# Force rebuild existing data
python -m data_synthesis.main_aggregation --rebuild

# Skip individual processing and only aggregate
python -m data_synthesis.main_aggregation --skip_individual
```
