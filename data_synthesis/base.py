#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   base.py
@Time    :   2025/08/02 12:04:00
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Base interfaces and classes for data synthesis module
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class MetadataTracker:
    """Tracks metadata for data generation and processing"""

    def __init__(self, design_name: str):
        self.design_name = design_name
        self.generation_timestamp = datetime.now().isoformat()
        self.data_counts = {}

    def update_data_count(self, data_type: str, count: int):
        """Update count for a specific data type"""
        self.data_counts[data_type] = count

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "design_name": self.design_name,
            "generation_timestamp": self.generation_timestamp,
            "data_counts": self.data_counts,
        }

    def save_to_file(self, output_dir: Path):
        """Save metadata to JSON file"""
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Metadata saved to {metadata_path}")


class DataProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, workspace: Path, output_dir: Path):
        self.workspace = workspace
        self.output_dir = output_dir
        self.metadata_tracker = None

    @abstractmethod
    def process(self) -> Dict[str, Any]:
        """Process data and return results"""
        pass

    @abstractmethod
    def validate_input(self) -> bool:
        """Validate input data"""
        pass

    def ensure_output_dir(self):
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class DataGenerator(ABC):
    """Abstract base class for data generators"""

    def __init__(self, design_name: str, output_dir: Path):
        self.design_name = design_name
        self.output_dir = output_dir
        self.metadata_tracker = MetadataTracker(design_name)

    @abstractmethod
    def generate_data_types(self) -> Dict[str, pd.DataFrame]:
        """Generate all data types as Parquet-ready DataFrames"""
        pass

    def save_parquet_files(
        self, data_dict: Dict[str, pd.DataFrame], enable_detailed_logs: bool = True
    ):
        """Save data as Parquet files with proper structure and detailed logging"""
        import time

        self.ensure_output_dir()

        for data_type, df in data_dict.items():
            if df is not None and not df.empty:
                parquet_path = self.output_dir / f"{data_type}.parquet"

                # Time the saving operation
                start_time = time.time()
                df.to_parquet(parquet_path, engine="pyarrow", index=False)
                save_duration = time.time() - start_time

                # Get file size
                file_size = parquet_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                # Update metadata
                self.metadata_tracker.update_data_count(data_type, len(df))

                # Detailed logging
                if enable_detailed_logs:
                    logging.info(
                        f"Saved {len(df)} records to {parquet_path} "
                        f"(duration: {save_duration:.2f}s, size: {file_size_mb:.2f}MB)"
                    )
                else:
                    logging.info(f"Saved {len(df)} records to {parquet_path}")
            else:
                logging.warning(f"No data generated for type: {data_type}")

        # Save metadata
        self.metadata_tracker.save_to_file(self.output_dir)

    def ensure_output_dir(self):
        """Ensure output directory exists"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_existing_metadata(self) -> Optional[Dict[str, Any]]:
        """Load existing metadata if available"""
        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None


class ParquetDataLoader:
    """Utility class for loading Parquet files efficiently"""

    @staticmethod
    def load_design_data(
        design_path: Path, data_types: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load specific data types from a design directory"""
        data = {}

        if not design_path.exists():
            logging.warning(f"Design directory not found: {design_path}")
            return data

        # Load metadata
        metadata_path = design_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                logging.info(
                    f"Loading design: {metadata.get('design_name', 'unknown')}"
                )

        # Discover available data types if not specified
        if data_types is None:
            data_types = []
            for file_path in design_path.iterdir():
                if file_path.suffix == ".parquet":
                    data_types.append(file_path.stem)

        # Load requested data types
        for data_type in data_types:
            parquet_path = design_path / f"{data_type}.parquet"
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    data[data_type] = df
                    logging.info(f"Loaded {len(df)} records from {data_type}.parquet")
                except Exception as e:
                    logging.error(f"Error loading {parquet_path}: {e}")
            else:
                logging.warning(f"Data type {data_type} not found in {design_path}")

        return data

    @staticmethod
    def validate_parquet_structure(design_dir: Path) -> bool:
        """Validate that design directory has proper Parquet structure"""
        expected_data_types = [
            "net_seqs",
            "pin2pin_pattern_seqs",
            "pin2pin_loc_seqs",
            "design_graph",
        ]
        
        # Check metadata file
        metadata_path = design_dir / "metadata.json"
        if not metadata_path.exists():
            logging.error(f"Missing metadata.json in {design_dir}")
            return False

        # Check for at least one data type
        found_data_types = []
        for data_type in expected_data_types:
            parquet_path = design_dir / f"{data_type}.parquet"
            if parquet_path.exists():
                found_data_types.append(data_type)

        if not found_data_types:
            logging.error(f"No valid data types found in {design_dir}")
            return False

        logging.info(f"Valid structure found with data types: {found_data_types}")
        return True


class ConfigurationManager:
    """Simplified configuration manager for data synthesis operations"""

    def __init__(
        self,
        max_turn_num: int = 4,
        rebuild: bool = False,
        enable_dataset_logs: bool = True,
        data_types: Optional[List[str]] = None,
    ):
        """
        Initialize configuration manager with simplified settings

        Args:
            max_turn_num: Maximum number of turns for pattern processing (default: 4)
            rebuild: Force rebuild of existing data (default: False)
            enable_dataset_logs: Enable detailed logging for dataset processing (default: True)
            data_types: List of data types to process (defaults to all types)
        """
        self.max_turn_num = max_turn_num
        self.rebuild = rebuild
        self.enable_dataset_logs = enable_dataset_logs
        self.data_types = data_types or [
            "net_seqs",
            "pin2pin_pattern_seqs",
            "pin2pin_loc_seqs",
            "design_graph",
        ]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self, key, default)
