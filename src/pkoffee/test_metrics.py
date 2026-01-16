"""Test the model evaluation metrics for assessing fit quality."""

import pytest
import numpy as np

def test_import():
	from pkoffee import metrics

def test_size_mismatch_valid():
    from pkoffee.metrics import check_size_match
    a = np.array(object=[1, 2, 3])
    b = np.array(object=[2, 2, 4])
    check_size_match(a, b)
