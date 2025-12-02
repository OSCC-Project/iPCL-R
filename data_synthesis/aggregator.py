#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   aggregator.py
@Time    :   2025/08/02 12:03:35
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Dataset aggregator for creating HuggingFace DatasetDict from design data
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict

from .base import ConfigurationManager, ParquetDataLoader


class DatasetAggregator:
    """Aggregates individual design data into consolidated HuggingFace DatasetDict"""

    def __init__(
        self,
        base_data_dir: Path,
        output_dir: Path,
        config: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize dataset aggregator

        Args:
            base_data_dir: Base directory containing individual design data
            output_dir: Output directory for consolidated dataset
            config: Configuration manager
        """
        self.base_data_dir = base_data_dir
        self.output_dir = output_dir
        self.config = config or ConfigurationManager()

    def aggregate_datasets(
        self,
        designs_to_include: Optional[List[str]] = None,
        types_to_include: Optional[List[str]] = None,
    ) -> DatasetDict:
        """
        Aggregate individual design datasets into consolidated DatasetDict

        Args:
            designs_to_include: List of design names to include
            types_to_include: List of data types to include

        Returns:
            HuggingFace DatasetDict with structure: {design: {type: Dataset}}
        """
        logging.info("Starting dataset aggregation...")

        # Discover available designs if not specified
        if designs_to_include is None:
            designs_to_include = self._discover_designs()

        # Use default data types if not specified
        if types_to_include is None:
            types_to_include = self.config.get(
                "data_types",
                [
                    "net_seqs",
                    "pin2pin_pattern_seqs",
                    "pin2pin_loc_seqs",
                    "design_graph",
                ],
            )

        logging.info(
            f"Aggregating {len(designs_to_include)} designs with types: {types_to_include}"
        )

        # Load and aggregate data
        aggregated_data = {}

        for design_name in designs_to_include:
            design_dir = self.base_data_dir / design_name

            if not ParquetDataLoader.validate_parquet_structure(design_dir):
                logging.warning(f"Invalid structure for design {design_name}, skipping")
                continue

            design_data = self._load_design_data(design_dir, types_to_include)
            if design_data:
                aggregated_data[design_name] = design_data
                logging.info(f"Loaded design: {design_name}")
            else:
                logging.warning(f"No data loaded for design: {design_name}")

        if not aggregated_data:
            logging.error("No design data loaded")
            return DatasetDict()

        # Convert to DatasetDict
        dataset_dict = self._create_dataset_dict(aggregated_data)

        # Save aggregated dataset
        self._save_dataset_dict(dataset_dict)

        # Save aggregation metadata
        self._save_aggregation_metadata(designs_to_include, types_to_include)

        logging.info(f"Successfully aggregated {len(aggregated_data)} designs")
        return dataset_dict

    def _discover_designs(self) -> List[str]:
        """Discover available designs in base data directory"""
        designs = []

        if not self.base_data_dir.exists():
            logging.error(f"Base data directory not found: {self.base_data_dir}")
            return designs

        for item_dir in self.base_data_dir.iterdir():
            if item_dir.is_dir():
                # Check if it has valid Parquet structure
                if ParquetDataLoader.validate_parquet_structure(item_dir):
                    designs.append(item_dir.name)

        logging.info(f"Discovered {len(designs)} valid designs: {designs}")
        return designs

    def _load_design_data(
        self, design_dir: Path, types_to_include: List[str]
    ) -> Dict[str, Dataset]:
        """Load design data and convert to HuggingFace Datasets"""
        # Load Parquet data
        parquet_data = ParquetDataLoader.load_design_data(design_dir, types_to_include)

        if not parquet_data:
            return {}

        # Convert to HuggingFace Datasets
        design_datasets = {}
        for data_type, df in parquet_data.items():
            if df is not None and not df.empty:
                try:
                    # Convert DataFrame to Dataset
                    dataset = Dataset.from_pandas(df)
                    design_datasets[data_type] = dataset
                    logging.debug(
                        f"Converted {data_type} to Dataset with {len(dataset)} records"
                    )
                except Exception as e:
                    logging.error(f"Error converting {data_type} to Dataset: {e}")

        return design_datasets

    def _create_dataset_dict(
        self, aggregated_data: Dict[str, Dict[str, Dataset]]
    ) -> DatasetDict:
        """Create nested DatasetDict structure"""
        nested_dict = {}

        for design_name, design_datasets in aggregated_data.items():
            if design_datasets:
                nested_dict[design_name] = DatasetDict(design_datasets)

        return DatasetDict(nested_dict)

    def _save_dataset_dict(self, dataset_dict: DatasetDict):
        """Save DatasetDict to disk"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as HuggingFace dataset format
        dataset_path = self.output_dir / "aggregated_dataset"
        dataset_dict.save_to_disk(dataset_path)
        logging.info(f"Saved DatasetDict to {dataset_path}")

        # Also save dataset info as JSON for easy inspection
        info_path = self.output_dir / "dataset_info.json"
        dataset_info = self._get_dataset_info(dataset_dict)
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        logging.info(f"Saved dataset info to {info_path}")

    def _save_aggregation_metadata(
        self, designs_included: List[str], types_included: List[str]
    ):
        """Save metadata about the aggregation process"""
        metadata = {
            "designs_included": designs_included,
            "types_included": types_included,
            "total_designs": len(designs_included),
            "aggregation_timestamp": pd.Timestamp.now().isoformat(),
            "base_data_dir": str(self.base_data_dir),
            "output_dir": str(self.output_dir),
        }

        metadata_path = self.output_dir / "aggregation_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved aggregation metadata to {metadata_path}")

    def _get_dataset_info(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """Get information about the dataset structure"""
        info = {"total_designs": len(dataset_dict), "designs": {}}

        for design_name, design_dataset_dict in dataset_dict.items():
            design_info = {
                "data_types": list(design_dataset_dict.keys()),
                "total_records_by_type": {},
            }

            for data_type, dataset in design_dataset_dict.items():
                design_info["total_records_by_type"][data_type] = len(dataset)

            info["designs"][design_name] = design_info

        return info

    def create_flat_corpus(
        self,
        dataset_dict: DatasetDict,
        data_type: str = "net_seqs",
        output_path: Optional[Path] = None,
    ) -> Dataset:
        """
        Create a flattened corpus dataset from the aggregated data

        Args:
            dataset_dict: Aggregated DatasetDict
            data_type: Data type to flatten (e.g., 'net_seqs')
            output_path: Optional path to save flattened corpus

        Returns:
            Flattened Dataset suitable for tokenization
        """
        logging.info(f"Creating flat corpus from {data_type}")

        all_records = []

        for design_name, design_datasets in dataset_dict.items():
            if data_type in design_datasets:
                dataset = design_datasets[data_type]

                # Add design information to each record
                for record in dataset:
                    enhanced_record = {**record, "source_design": design_name}
                    all_records.append(enhanced_record)

        if not all_records:
            logging.warning(f"No records found for data type: {data_type}")
            return Dataset.from_list([])

        # Create flattened dataset
        flat_dataset = Dataset.from_list(all_records)

        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            flat_dataset.save_to_disk(output_path)
            logging.info(f"Saved flat corpus to {output_path}")

        logging.info(f"Created flat corpus with {len(flat_dataset)} records")
        return flat_dataset

    def get_statistics(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """Get comprehensive statistics about the aggregated dataset"""
        stats = {
            "total_designs": len(dataset_dict),
            "data_types": set(),
            "total_records_by_type": defaultdict(int),
            "design_breakdown": {},
        }

        for design_name, design_datasets in dataset_dict.items():
            design_stats = {
                "data_types": list(design_datasets.keys()),
                "records_by_type": {},
            }

            for data_type, dataset in design_datasets.items():
                stats["data_types"].add(data_type)
                record_count = len(dataset)
                stats["total_records_by_type"][data_type] += record_count
                design_stats["records_by_type"][data_type] = record_count

            stats["design_breakdown"][design_name] = design_stats

        stats["data_types"] = list(stats["data_types"])
        stats["total_records_by_type"] = dict(stats["total_records_by_type"])

        return stats
