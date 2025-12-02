#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   feature_gen.py
@Time    :   2025/01/23 23:04:03
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   feature generator for layout_patch_represent
"""

import sys
from pathlib import Path


def setup_aieda() -> bool:
    """setup aieda"""
    try:
        project_root = Path(__file__).resolve().parent.parent
        aieda_path = project_root / "third_party" / "aieda"

        if str(aieda_path) not in sys.path:
            sys.path.append(str(aieda_path))

    except Exception as e:
        print(f"Failed to set up AIEDA: {e}")
        return False
    return True


class LmT28FeatureGenerate:
    """feature generate"""

    def __init__(self, dir_workspace: Path, design: str):
        self.dir_workspace = dir_workspace
        self.design = design

    def get_lm_graph(self):
        try:
            if setup_aieda():
                from aieda.data.vectors import DataVectors
                from aieda.third_party.tools import workspace_create
            else:
                print("Failed to set up AIEDA")
                return None
        except Exception as e:
            print(f"Failed to set up AIEDA: {e}")
            return None

        workspace = workspace_create(
            directory=str(self.dir_workspace), design=self.design
        )
        lm_graph = DataVectors(workspace=workspace)

        return lm_graph.load_nets()
