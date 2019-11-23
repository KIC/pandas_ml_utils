from unittest import TestCase

import numpy as np

from pandas_ml_utils.utils import unfold_parameter_space, KFoldBoostRareEvents, ReScaler
import subprocess
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")


class TestMakeDocs(TestCase):

    def test_make_clean_html(self):
        """when"""
        docs_process = subprocess.run(["make", "clean", "html"], cwd=path, capture_output=True)

        """expect"""
        print(docs_process)
        self.assertEqual(docs_process.returncode, 0)