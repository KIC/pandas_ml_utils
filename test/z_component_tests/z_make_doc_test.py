import os
import pytest
import subprocess
from unittest import TestCase

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs")


class TestMakeDocs(TestCase):

    @pytest.mark.skipif(os.environ.get('USER') != 'kic', reason="do not run this test in github actions")
    def test_make_clean_html(self):
        """when"""
        docs_process = subprocess.run(["make", "clean", "html"], cwd=path, capture_output=True)

        """expect"""
        print(docs_process)
        self.assertEqual(docs_process.returncode, 0)