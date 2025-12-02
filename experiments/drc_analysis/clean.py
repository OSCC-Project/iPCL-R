#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   clean.py
@Time    :   2025/09/06 17:30:00
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Clean up summary generated files and DRC report files
"""

import argparse
import logging
from pathlib import Path

from flow.utils import setup_logging


class DRCFilesCleaner:
    """Cleaner for DRC summary files and report files."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.summary_files = [
            "drc_distribution.csv",
            "drc_distribution_summary.md",
            "drc_eco_distribution.csv",
            "drc_eco_distribution_summary.md",
        ]
        self.drc_report_patterns = ["drc*.rpt*"]

    def clean_summary_files(self, base_dir: Path) -> None:
        """Clean summary generated files in each design/meta_type directory."""
        if not base_dir.exists():
            logging.warning(f"Base directory does not exist: {base_dir}")
            return

        logging.info(f"Cleaning summary files in: {base_dir}")
        cleaned_count = 0

        # Traverse all design directories
        for design_dir in base_dir.iterdir():
            if not design_dir.is_dir():
                continue

            logging.info(f"  Processing design: {design_dir.name}")

            # Traverse all meta_type directories (GT, POST, PRED)
            for meta_type_dir in design_dir.iterdir():
                if not meta_type_dir.is_dir():
                    continue

                logging.info(f"    Processing meta_type: {meta_type_dir.name}")

                # Clean summary files
                for filename in self.summary_files:
                    file_path = meta_type_dir / filename
                    if file_path.exists():
                        if self.dry_run:
                            logging.info(f"      - Would remove: {file_path}")
                            cleaned_count += 1
                        else:
                            try:
                                file_path.unlink()
                                logging.info(f"      ✓ Removed: {file_path}")
                                cleaned_count += 1
                            except Exception as e:
                                logging.error(
                                    f"      ✗ Failed to remove {file_path}: {e}"
                                )
                    else:
                        logging.warning(f"      - Not found: {filename}")

        logging.info(
            f"Summary files cleanup completed. {'Would remove' if self.dry_run else 'Removed'} {cleaned_count} files."
        )

    def clean_drc_reports(self, base_dir: Path) -> None:
        """Clean DRC report files (drc*.rpt*) in each design/meta_type directory."""
        if not base_dir.exists():
            logging.warning(f"Base directory does not exist: {base_dir}")
            return

        logging.info(f"Cleaning DRC report files in: {base_dir}")
        cleaned_count = 0

        # Traverse all design directories
        for design_dir in base_dir.iterdir():
            if not design_dir.is_dir():
                continue

            logging.info(f"  Processing design: {design_dir.name}")

            # Traverse all meta_type directories (GT, POST, PRED)
            for meta_type_dir in design_dir.iterdir():
                if not meta_type_dir.is_dir():
                    continue

                logging.info(f"    Processing meta_type: {meta_type_dir.name}")

                # Find and clean DRC report files
                drc_files = list(meta_type_dir.glob("drc*.rpt*"))
                for drc_file in drc_files:
                    if self.dry_run:
                        logging.info(f"      - Would remove: {drc_file}")
                        cleaned_count += 1
                    else:
                        try:
                            drc_file.unlink()
                            logging.info(f"      ✓ Removed: {drc_file}")
                            cleaned_count += 1
                        except Exception as e:
                            logging.error(f"      ✗ Failed to remove {drc_file}: {e}")

                if not drc_files:
                    logging.warning("      - No DRC report files found")

        logging.info(
            f"DRC report files cleanup completed. {'Would remove' if self.dry_run else 'Removed'} {cleaned_count} files."
        )

    def clean_timing_design_files(self, target_dir: Path) -> int:
        """Clean timing design related files (including postRoute files)"""
        removed_count = 0
        logging.info(f"Cleaning timing design files (*postRoute*) in: {target_dir}")

        if not target_dir.exists():
            logging.warning(f"Directory {target_dir} does not exist")
            return removed_count

        for design_dir in target_dir.iterdir():
            if design_dir.is_dir():
                logging.info(f"  Processing design: {design_dir.name}")

                # Directly search for the postRoute file in the design directory.
                postroute_files = list(design_dir.glob("*postRoute*"))
                if postroute_files:
                    logging.info(
                        f"    Found {len(postroute_files)} postRoute files in design directory"
                    )
                    for file_path in postroute_files:
                        if self.dry_run:
                            logging.info(f"      - Would remove: {file_path}")
                        else:
                            logging.info(f"      - Removing: {file_path}")
                            file_path.unlink()
                        removed_count += 1

                # Check the meta_type subdirectory.
                for meta_type_dir in design_dir.iterdir():
                    if meta_type_dir.is_dir():
                        postroute_files = list(meta_type_dir.glob("*postRoute*"))

                        if postroute_files:
                            logging.info(
                                f"    Processing meta_type: {meta_type_dir.name}"
                            )
                            for file_path in postroute_files:
                                if self.dry_run:
                                    logging.info(f"      - Would remove: {file_path}")
                                else:
                                    logging.info(f"      - Removing: {file_path}")
                                    file_path.unlink()
                                removed_count += 1

        logging.info(
            f"Timing design files cleanup completed. Removed {removed_count} files."
        )
        return removed_count

    def clean_all(self, base_dir: Path) -> None:
        """Clean summary files, DRC report files, and timing design files."""
        logging.info("=" * 60)
        logging.info("Starting cleanup process...")
        logging.info("=" * 60)

        # Clean summary files first
        self.clean_summary_files(base_dir)
        logging.info()

        # Then clean DRC report files
        self.clean_drc_reports(base_dir)
        logging.info()

        # Finally clean timing design files
        self.clean_timing_design_files(base_dir)

        logging.info("=" * 60)
        logging.info("Cleanup process completed!")
        logging.info("=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up DRC summary files and report files"
    )
    parser.add_argument(
        "--base",
        "-b",
        default=Path("/data2/project_share/liweiguo/rt_gen/output"),
        type=Path,
        help="Base directory containing design folders (default: /data2/project_share/liweiguo/rt_gen/output)",
    )
    parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Clean only summary files (CSV and MD files)",
    )
    parser.add_argument(
        "--reports-only",
        "-r",
        action="store_true",
        help="Clean only DRC report files (drc*.rpt*)",
    )
    parser.add_argument(
        "--timing-design-only",
        "-t",
        action="store_true",
        help="Clean only timing design files (*postRoute*)",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be cleaned without actually removing files",
    )
    return parser.parse_args()


def main():
    setup_logging()

    """Main entry point for the cleanup script."""
    args = parse_arguments()
    cleaner = DRCFilesCleaner(dry_run=args.dry_run)

    if args.dry_run:
        logging.info("DRY RUN MODE - No files will be actually removed")
        logging.info()

    if args.summary_only:
        logging.info("Cleaning summary files only...")
        cleaner.clean_summary_files(args.base)
    elif args.reports_only:
        logging.info("Cleaning DRC report files only...")
        cleaner.clean_drc_reports(args.base)
    elif args.timing_design_only:
        logging.info("Cleaning timing design files only...")
        cleaner.clean_timing_design_files(args.base)
    else:
        logging.info("Cleaning all files...")
        cleaner.clean_all(args.base)


if __name__ == "__main__":
    main()
