#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_tokenization.py
@Time    :   2025/08/01 11:16:22
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Command-line launcher for Stage 1 tokenization pipeline using FlowConfig,
             handles UnifiedTokenizer initialization and routing pattern sequence
             tokenization with progress reporting
"""

import argparse
import sys
import time
from pathlib import Path

from flow.config import FlowConfig
from flow.tokenization import TokenizationPipeline
from flow.utils import setup_logging


def main():
    setup_logging()
    """Main entry point for tokenization launcher"""
    parser = argparse.ArgumentParser(
        description="Launch tokenization pipeline using FlowConfig"
    )
    parser.add_argument(
        "--flow-config", type=str, required=True, help="Path to FlowConfig JSON file"
    )

    args = parser.parse_args()

    try:
        # Load FlowConfig from JSON file
        print(f"🔧 Loading configuration from: {args.flow_config}")
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))

        # Extract tokenization stage configuration
        tokenization_config = flow_config.tokenization
        dataset_config = flow_config.dataset

        print("🚀 Starting tokenization pipeline")
        print(f"   Algorithm: {tokenization_config.workflow.tokenizer_algorithm}")
        print(f"   Target Vocab size: {tokenization_config.workflow.target_vocab_size}")
        if dataset_config.use_hub():
            print(
                f"   Data source: {dataset_config.hub_id} (split: {dataset_config.train_split})"
            )
        else:
            print(
                f"   Data source: {dataset_config.local_path_for_split(dataset_config.train_split)}"
            )
        print(f"   Output: {tokenization_config.paths.token_dataset_dir}")
        print()

        # Create and run tokenization pipeline
        pipeline = TokenizationPipeline(flow_config)

        start_time = time.time()
        results = pipeline.run_tokenization()
        end_time = time.time()

        # Report results
        execution_time = end_time - start_time
        print("✅ Tokenization completed successfully!")
        print(f"   Execution time: {execution_time:.2f} seconds")
        print(f"   Tokenizer saved to: {tokenization_config.paths.tokenizer_save_dir}")
        print(f"   Dataset saved to: {tokenization_config.paths.token_dataset_dir}")

        if "metadata" in results:
            metadata = results["metadata"]
            print("   Dataset info:")
            if "total_samples" in metadata:
                print(f"     Total samples: {metadata['total_samples']}")
            if "vocabulary_size" in metadata:
                print(f"     Vocabulary size: {metadata['vocabulary_size']}")
            if "avg_sequence_length" in metadata:
                print(
                    f"     Avg sequence length: {metadata['avg_sequence_length']:.1f}"
                )

        print(f"   Metadata saved to: {tokenization_config.paths.output_metadata_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: Configuration file not found: {e}")
        return 1
    except KeyError as e:
        print(f"❌ Error: Missing configuration key: {e}")
        return 1
    except Exception as e:
        print(f"❌ Tokenization failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
