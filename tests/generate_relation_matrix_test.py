import numpy as np
import pytest
from src.placket.generate_relation_matrix.py import Relation_Table

def test_matrix_construction():
    # Setup
    model = HamiltonianBuilder(size=2)
    
    # Run
    result = model.compute_table()
    
    # Check
    expected = np.array([[0, 0.8+0.5j], [0.8-0.5j, 0]])
    
    # NumPy testing works great inside pytest functions
    np.testing.assert_allclose(result, expected, rtol=1e-5)