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
import static.unit_cell as consts
import importlib

from scipy.optimize import minimize_scalar, brentq
from sympy import symbols, Matrix, pprint, nsimplify

from src.placket.generate_relation_matrix import Relation_Table
importlib.reload(consts)

def what_x_gives_y(wanted_value, func, x_min=-1000, x_max=1000):
# Minimize |func(x) - wanted_value|
    result = minimize_scalar(
        lambda x: abs(func(x) - wanted_value),
        bounds=(x_min, x_max),
        method='bounded'
    )
    return result.x

def test_matrix_construction():
    # Setup
    table = consts.conduction_electron_cell.table1
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
    Number_points_integrate_over = 2
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
    def real_phase_func_0(k_x,k_y):
        energy = np.zeros(1)
        energy[0] = 2 * np.cos(k_x) + 4 * np.cos(0.5 * k_x) * np.cos(np.sqrt(3)/2 * k_y)
        return energy 
    
    class_table_0 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))
    class_phase_0 = class_table_0.create_momentum_transform()
    
    class_table_2 = Relation_Table(2, total_bonds = 6, basis_bonds = (2,3), basis_unit_cell = ((2,-1),(0,2))) #used to be basis_unit_cell = ((1,2),(-2,2)))
    class_phase_2 = class_table_2.create_momentum_transform()
    
    # print(f"{class_table_0.view_momentum_matrix()=}")
    # print(f"{class_table_2.view_momentum_matrix()=}")
    
    testing.append((class_phase_0, real_phase_func_0 ))
    testing.append((class_phase_2, real_phase_2))
    
    #run over all tests
    for idx, test in enumerate(testing):
        #functions for this test, new every test
        class_phase_func, real_phase_func = test
        #Energy counters for this test, reset every test
        total_energy_class = 0
        total_energy_real = 0
        
        #each tests we will integrate over the full BZ
        for k1 in k1_grid:
            for k2 in k2_grid:
                #class phase calculations
                class_phase_matrix = class_phase_func(k1,k2)
                eigenvalues_class = np.linalg.eigvalsh(class_phase_matrix)
                if not np.allclose(np.imag(eigenvalues_class), 0, atol=1e-10):
                    logger.warning(f"Warning: Large imaginary components found at k=({k1}, {k2})! Matrix might not be Hermitian.")
            
                total_energy_class += np.sum(np.real(eigenvalues_class)[np.real(eigenvalues_class) < 0 ])
                
                #hand phase calculations
                real_phase_matrix = real_phase_func(k1,k2)
                if real_phase_matrix.shape == (1,):
                    energy_val = np.real(real_phase_matrix[0])
                    if energy_val < 0:
                        total_energy_real += energy_val
                else:
                    eigenvalues_real = np.linalg.eigvalsh(real_phase_matrix)
                    total_energy_real += np.sum(eigenvalues_real[eigenvalues_real < 0 ])
               
                #check that the matrices are close in value
                # assert np.allclose( real_phase_matrix, class_phase_matrix, rtol=1e-9, atol=1e-9)
                
                #print out the matrix info for the point
                # print(f"######{float(k1)=}, {float(k2)=}#######")
                # print(f"Class:\n{class_phase_matrix}")
                # print(f"Hand:\n{real_phase_matrix}")
                # print(f"{total_energy_class=}")
                # print(f"{total_energy_real=}")
                # print("#############")
                
        ##### ONCE WE FINISH ADDING UP ALL THE POINTS IN THE BZ #######
        #we can normalize our result
        ground_state_energy_class_normalzied = total_energy_class / (Number_points_integrate_over ** 2)
        ground_state_energy_real_normalzied = total_energy_real  / (Number_points_integrate_over ** 2)
        #we can now compare our resutls
        print(f"{idx=}")
        print(f"{ground_state_energy_class_normalzied=}")
        print(f"{ground_state_energy_real_normalzied=}")
        assert np.isclose(ground_state_energy_class_normalzied, ground_state_energy_real_normalzied, atol=1e-8)        
        ################ END ALL TESTS ################
    
def test_mu_value():
    #Controls precsion of calculation
    Number_points_integrate_over = 100
    #Set up grid integration
    k1_grid = np.linspace(0,1,Number_points_integrate_over)
    k2_grid = np.linspace(0,1,Number_points_integrate_over)
    testing = []
    results = []

    #Define all known matrices to compare against

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
    def real_phase_func_0(k_x,k_y):
        energy = np.zeros(1)
        energy[0] = 2 * np.cos(k_x) + 4 * np.cos(0.5 * k_x) * np.cos(np.sqrt(3)/2 * k_y)
        return energy 
        
    class_table_0 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))
    class_phase_0 = class_table_0.create_momentum_transform()
    
    class_table_2 = Relation_Table(2 , total_bonds = 6, basis_bonds = (2,3), basis_unit_cell = ((2,-1),(0,2))) #used to be basis_unit_cell = ((1,2),(-2,2)))
    class_phase_2 = class_table_2.create_momentum_transform()
    
    print(f"{class_table_0.view_momentum_matrix()=}")
    print(f"{class_table_2.view_momentum_matrix()=}")
    
    # testing.append((class_phase_0, real_phase_func_0,1))
    testing.append((class_phase_2, real_phase_2,2))
    for test in testing:
        class_phase_func, real_phase_func, cell_size = test
        def compute_mu(mu, phase_func, unit_cell_size):
            total_sites = 0  # Reset on every call
            for k1 in k1_grid:
                for k2 in k2_grid:
                    matrix = phase_func(k1, k2)
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    if not np.allclose(np.imag(eigenvalues), 0, atol=1e-10):
                        logger.warning(f"Large imaginary components at k=({k1}, {k2})")
                    total_sites += np.sum(np.real(eigenvalues) < mu)  # vectorized
            density = total_sites / (len(k1_grid) * len(k2_grid) * unit_cell_size)
            return density
        
        wanted_density = 0.5
        mu_class = brentq(
            lambda mu: compute_mu(mu, class_phase_func,cell_size) - wanted_density,
            a=-10,   # lower bound for mu search
            b=10     # upper bound for mu search
        )
        mu_real = brentq(
            lambda mu: compute_mu(mu, real_phase_func,cell_size) - wanted_density,
            a=-10,   # lower bound for mu search
            b=10     # upper bound for mu search
        )
        print(f"*"*60)
        print(f"This is what I set the density to:{wanted_density}")
        print(f"THIS IS THE MU VALUE FROM MY CLASS MATRIX: {mu_class:.3f}")
        print(f"THIS IS THE MU VALUE FROM MY HAND MATRIX: {mu_real:.3f}")
        print(f"*"*60)
        assert np.isclose(mu_class,mu_real, atol=1e-9)
   
def test_coupling_up():
    #table 0 test
    class_table_0 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))
    g = symbols('g')
    #get your matrix
    momentum_spin_up_coupled_matrix_0 = class_table_0.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=1)
    #'evaluate' the matrix using symbols so you can see what it looks like
    k1,k2 = symbols('k1 k2')
    final_matrix = momentum_spin_up_coupled_matrix_0(k1,k2)
    
    print(f"Spin Up Conduction Band ↑Pattern 1↑ \n\n") 
    # Round and Simplify the Sympy Matrix and pretty-print it afterwards!
    clean_matrix = nsimplify(final_matrix, tolerance=1e-5)
    pprint(Matrix(clean_matrix)) 
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))

    #table 1 test
    class_table_1 = Relation_Table(2, total_bonds = 6, basis_bonds=(2,3), basis_unit_cell = ((2,-1),(0,2)))
    #get your matrix
    momentum_spin_up_coupled_matrix_1 = class_table_1.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=1)
    #'evaluate' the matrix using symbols so you can see what it looks like
    k1,k2 = symbols('k1 k2')
    final_matrix = momentum_spin_up_coupled_matrix_1(k1,k2)
    
    print(f"Spin Up Conduction Band ↑Pattern 2↑ \n\n") 
    # Round and Simplify the Sympy Matrix and pretty-print it afterwards!
    clean_matrix = nsimplify(final_matrix, tolerance=1e-5)
    pprint(Matrix(clean_matrix))
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))
 

def test_coupling_down():
    
    #table 0 test
    class_table_0 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))
    g = symbols('g')
    #get your matrix
    momentum_spin_up_coupled_matrix_0 = class_table_0.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=-1)
    #'evaluate' the matrix using symbols so you can see what it looks like
    k1,k2 = symbols('k1 k2')
    final_matrix = momentum_spin_up_coupled_matrix_0(k1,k2)
    
    print(f"Spin Down Conduction Band ↓Pattern 1↓ \n\n") 
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))
    # Round and Simplify the Sympy Matrix and pretty-print it afterwards!
    clean_matrix = nsimplify(final_matrix, tolerance=1e-5)
    pprint(Matrix(clean_matrix))

    #table 1 test
    class_table_1 = Relation_Table(2, total_bonds = 6, basis_bonds=(2,3), basis_unit_cell = ((2,-1),(0,2)))
    #get your matrix
    momentum_spin_up_coupled_matrix_1 = class_table_1.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=-1)
    #'evaluate' the matrix using symbols so you can see what it looks like
    k1,k2 = symbols('k1 k2')
    final_matrix = momentum_spin_up_coupled_matrix_1(k1,k2)
    
    print(f"Spin Down Conduction Band ↓Pattern 2↓ \n\n") 
    # Round and Simplify the Sympy Matrix and pretty-print it afterwards!
    clean_matrix = nsimplify(final_matrix, tolerance=1e-5)
    pprint(Matrix(clean_matrix))
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))

if __name__ == "__main__":
    # test_matrix_construction()
    test_coupling_up()
    test_coupling_down()