import sys 
import os
# 1. Get the path to 'tests'
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the path to 'Kitaev-Diagnolization' (parent of tests)
project_root = os.path.dirname(current_dir)
# 3. Add it to the system path
sys.path.insert(0, project_root)

import numpy as np
import pytest
from src.placket.generate_relation_matrix import Relation_Table
import static.unit_cell as consts
import importlib
importlib.reload(consts)

def test_matrix_construction():
    # Setup
    table = consts.params.table1
    model_pahse_2 = Relation_Table(neighbor_table = table, total_bonds = 6, basis_bonds = (2,3), basis_unit_cell = ((2,-1),(0,2))) #used to be basis_unit_cell = ((1,2),(-2,2)))
    
    # Run
    dcomposed_bond_1 = model_pahse_2._decompose_bond(2)
    decomposed_bond_2 = model_pahse_2._decompose_bond(3)
    decompose_bond_diffrent = model_pahse_2._decompose_bond(6)
    
    # Expected Value
    expected_decompsoed_bonds_1 = (1,0)
    expected_decompsoed_bonds_2 = (0,1)
    expected_decompose_bonds_diff = (0,-1)
    
    
    # NumPy testing works great inside pytest functions
    np.testing.assert_allclose(dcomposed_bond_1, expected_decompsoed_bonds_1, rtol=1e-5)
    np.testing.assert_allclose(decomposed_bond_2, expected_decompsoed_bonds_2, rtol=1e-5)
    np.testing.assert_allclose(decompose_bond_diffrent, expected_decompose_bonds_diff, rtol=1e-3, atol= 1e-3)
    
    final_table_model_phase_2 = model_pahse_2.create_momentum_transform()
    # print(f"Final table: {final_table_model_phase_2(1,2)} with k1,k2 = 1,2")
    return

#Testing actual numerical value against known coded tables
def test_matrix_evaluation():
    #Controls precsion of calculation
    Number_points_integrate_over = 1000
    #Set up grid integration
    k1_grid = np.linspace(0,1,Number_points_integrate_over)
    k2_grid = np.linspace(0,1,Number_points_integrate_over)
    
    #Define all known matrices to compare against
    testing = [] #insert everything in this as tuples (Class made matrix, Actual matrix)
    def real_phase_2(k1,k2):
        actual_phase_2 = np.zeros((2,2), dtype=complex)
        actual_phase_2[0,0] = np.exp(np.pi * 1j *k2) + np.exp(-1* np.pi * 1j * k2)
        
        actual_phase_2[0,1] =(
            np.exp(np.pi * 1j * (k1 - k2/2)) + np.exp(-1 *np.pi * 1j * (k1 - k2/2)) 
            + np.exp(np.pi * 1j * (k1 + k2/2)) + np.exp(-1 *np.pi * 1j * (k1 + k2/2)) 
        )
        actual_phase_2[1,0] =(
            np.exp(np.pi * 1j * (k1 - k2/2)) + np.exp(-1 *np.pi * 1j * (k1 - k2/2)) 
            + np.exp(np.pi * 1j * (k1 + k2/2)) + np.exp(-1 *np.pi * 1j * (k1 + k2/2)) 
        )
        actual_phase_2[1,1] = np.exp(np.pi * 1j *k2) + np.exp(-1* np.pi * 1j  *k2)
    
        return actual_phase_2
    class_table_2 = Relation_Table(neighbor_tabel = consts.params.table1 , total_bonds = 6, basis_bonds = (2,3), basis_unit_cell = ((2,-1),(0,2))) #used to be basis_unit_cell = ((1,2),(-2,2)))
    class_phase_2 = class_table_2.create_momentum_transform()
    testing.append((class_phase_2, real_phase_2))
    
    
    
    

    #run over all tests
    for test in testing:
        #functions for this test, new every test
        class_phase_func, real_phase_func = test
        #Energy counters for this test, reset every test
        total_energy_class = 0
        total_energy_real = 0
        
        #each tests we will integrate over the full BZ
        for k1 in k1_grid:
            for k2 in k2_grid:
                #get the matrix for this point in the BZ
                class_phase_matrix = class_phase_func(k1,k2)
                real_phase_matrix = real_phase_func(k1,k2)
                #diagnolize the matrix for this point in the BZ
                eigenvalues_class = np.linalg.eigvalsh(class_phase_matrix)
                eigenvalues_real = np.linalg.eigvalsh(real_phase_matrix)
                #sums only the positive values for this point in the BZ
                total_energy_class += np.sum(eigenvalues_class[eigenvalues_class < 0 ])
                total_energy_real += np.sum(eigenvalues_real[eigenvalues_real < 0 ])
        ##### ONCE WE FINISH ADDING UP ALL THE POINTS IN THE BZ #######
        #we can normalize our result
        ground_state_energy_class_normalzied = total_energy_class / (Number_points_integrate_over ** 2)
        ground_state_energy_real_normalzied = total_energy_real  / (Number_points_integrate_over ** 2)
        #we can now compare our resutls
        print(f"{ground_state_energy_class_normalzied=}")
        print(f"{ground_state_energy_real_normalzied=}")
        assert ground_state_energy_class_normalzied == ground_state_energy_real_normalzied
        
        ################ END ALL TESTS ################
    
    
    
    
if __name__ == "__main__":
    test_matrix_construction()