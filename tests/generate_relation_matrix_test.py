import sys 
import os
from typing import Literal, Any, Tuple, Dict, Optional
import logging
logger = logging.getLogger(__name__)
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
from sympy import symbols, Matrix,Array, pprint, nsimplify, lambdify

from src.placket.generate_relation_matrix import Relation_Table
from src.core.base_matrix import BaseMatrix
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
    dcomposed_bond_1 = model_pahse_2._decompose_bond_neighbor_basis(2)
    decomposed_bond_2 = model_pahse_2._decompose_bond_neighbor_basis(3)
    decompose_bond_diffrent = model_pahse_2._decompose_bond_neighbor_basis(6)
    
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
   
def symbolic_test_coupling_up():
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
 

def symbolic_test_coupling_down():
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


def numerical_test_coupling_up():
    #set up symbols so we can use them for all equations
    g,k1,k2 = symbols('g k1 k2')

    # 1. Define a large grid of input values
    k1_vals = np.linspace(-np.pi, np.pi, 1000)
    k2_vals = np.linspace(-np.pi, np.pi, 1000)
    g_val = 1.5 
    # 2. Pass the entire arrays directly into the compiled function
    # This computes 1,000 matrices instantly in C
    k1_meshgrid,k2_meshgrid = np.meshgrid(k1_vals,k2_vals)

    #######################################  table 1 test  #######################################
    class_table_1 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))  
    #get your symbolic matrix
    momentum_spin_up_coupled_matrix_1 = class_table_1.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=1)
    pre_symbolic_matrix_1 = momentum_spin_up_coupled_matrix_1(k1,k2)
    symbolic_matrix_1 = Array(pre_symbolic_matrix_1) #can be Matrix(pre_symbolic_matrix) if the input is a 2-d matrix, use Array if not sure

    #convert symbolic to a computational matrix with lambdify
    compute_matrix_1 = lambdify((k1, k2, g), symbolic_matrix_1, modules='numpy')
    #intilize the matrix with full meshgrids to create a "stack" of matrixs
    full_computational_matrix_1 = compute_matrix_1(k1_meshgrid, k2_meshgrid, g_val)
    
    #3. Format the matrix correctly so np.eiginvals can compute it properly
    #Make sure you moveaxis/transpose -> reshape! RESHAPING should always happen last for speed!!
    #aka transpose then flatten
    transposed_computation_matrix_1 = np.transpose(full_computational_matrix_1,axes=(2,3,0,1)) #puts all the actual values in the front. The row and column numbers are just basically metadata that tell np how to interpert this massive list of list of values
    flat_computation_matrix_1 = transposed_computation_matrix_1.reshape(-1,transposed_computation_matrix_1.shape[-2],transposed_computation_matrix_1.shape[-1]) #-1 means "figure out the rest" this keeps the row and columns and basically squishes the rest of the value into one list
    final_computational_stacked_matrix_1 = np.asarray(flat_computation_matrix_1, dtype=np.complex128) #convert to complex array just for consietensiy before passing it to the solver

    #4. do checks to make sure nothing went wrong
    if not np.allclose(final_computational_stacked_matrix_1, final_computational_stacked_matrix_1.conj().transpose(0, 2, 1), atol=1e-10):
        logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")

    # 5. THE EIGENVALUES & SUMMATION 
    eigenvalues = np.linalg.eigvalsh(final_computational_stacked_matrix_1)
    total_energy = np.sum(eigenvalues[eigenvalues < 0])

    print(f"\n######################################\nSpin Up Conduction Band ↑Pattern 1↑") 
    print(f"The shape of your final matrix: {final_computational_stacked_matrix_1.shape=}")
    print(f"The values in of your final matrix: {final_computational_stacked_matrix_1=}") 
    print(f"{total_energy=}\n######################################\n") 
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))

    #######################################  table 2 test  #######################################
    class_table_2 = Relation_Table(2, total_bonds = 6, basis_bonds=(2,3), basis_unit_cell = ((2,-1),(0,2)))
    
    #get your symbolic matrix
    momentum_spin_up_coupled_matrix_2 = class_table_2.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=1)
    pre_symbolic_matrix_2 = momentum_spin_up_coupled_matrix_2(k1,k2)
    symbolic_matrix_2 = Matrix(pre_symbolic_matrix_2) #You can use Array strictly for 2-d array's instead

    #convert symbolic -> numpy computational using lamdify
    compute_matrix_2 = lambdify((k1, k2, g), symbolic_matrix_2, modules='numpy')
    #intilize the matrix with full meshgrids to create a "stack" of usable matrixs
    full_computational_matrix_2 = compute_matrix_2(k1_meshgrid, k2_meshgrid, g_val)
    
    #3. Format the matrix correctly so np.eiginvals can compute it properly
    #Make sure you moveaxis/transpose -> reshape! RESHAPING should always happen last for speed!!
    #aka transpose then flatten
    transposed_computation_matrix_2 = np.transpose(full_computational_matrix_2,axes=(2,3,0,1)) #puts all the actual values that we squished into one list at the front, this is how np.eginvalues needs it. The row and column are just basically metadata that tell np how to interpert this massive list of values
    flat_computation_matrix_2 = transposed_computation_matrix_2.reshape(-1,transposed_computation_matrix_2.shape[-2],transposed_computation_matrix_2.shape[-1]) #-1 means "figure out the rest" this keeps the row and columns and basically squishes the rest of the value into one list
    final_computational_stacked_matrix_2 = np.asarray(flat_computation_matrix_2, dtype=np.complex128) #convert to complex array just for consietensiy before passing it to the solver

    #4. do checks to make sure nothing went wrong
    if not np.allclose(final_computational_stacked_matrix_2, final_computational_stacked_matrix_2.conj().transpose(0, 2, 1), atol=1e-4):
        logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")

    # 5. THE EIGENVALUES & SUMMATION 
    eigenvalues = np.linalg.eigvalsh(final_computational_stacked_matrix_2)
    total_energy = np.sum(eigenvalues[eigenvalues < 0])

    
    print(f"\n######################################\nSpin Up Conduction Band ↑Pattern 2↑")
    print(f"The shape of your final matrix: {final_computational_stacked_matrix_2.shape=}")
    print(f"The values in of your final matrix: {final_computational_stacked_matrix_2=}") 
    print(f"{total_energy=}\n######################################\n") 
    return None

def numerical_test_coupling_down():
    #set up symbols so we can use them for all equations
    g,k1,k2 = symbols('g k1 k2')

    # 1. Define a large grid of input values
    k1_vals = np.linspace(-np.pi, np.pi, 1000)
    k2_vals = np.linspace(-np.pi, np.pi, 1000)
    g_val = 1.5 
    # 2. Pass the entire arrays directly into the compiled function
    # This computes 1,000 matrices instantly in C
    k1_meshgrid,k2_meshgrid = np.meshgrid(k1_vals,k2_vals)


    #######################################  table 1 test  #######################################
    class_table_1 = Relation_Table(1, total_bonds=6, basis_bonds= (2,3), basis_unit_cell= ((1,0),(0,1)))  
    #get your symbolic matrix
    momentum_spin_up_coupled_matrix_1 = class_table_1.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=-1)
    pre_symbolic_matrix_1 = momentum_spin_up_coupled_matrix_1(k1,k2)
    symbolic_matrix_1 = Array(pre_symbolic_matrix_1) #can be Matrix(pre_symbolic_matrix) if the input is a 2-d matrix, use Array if not sure

    #convert symbolic to a computational matrix with lambdify
    compute_matrix_1 = lambdify((k1, k2, g), symbolic_matrix_1, modules='numpy')
    #intilize the matrix with full meshgrids to create a "stack" of matrixs
    full_computational_matrix_1 = compute_matrix_1(k1_meshgrid, k2_meshgrid, g_val)
    
    #3. Format the matrix correctly so np.eiginvals can compute it properly
    #Make sure you moveaxis/transpose -> reshape! RESHAPING should always happen last for speed!!
    #aka transpose then flatten
    transposed_computation_matrix_1 = np.transpose(full_computational_matrix_1,axes=(2,3,0,1)) #puts all the actual values that we squished into one list at the front, this is how np.eginvalues needs it. The row and column are just basically metadata that tell np how to interpert this massive list of values
    flat_computation_matrix_1 = transposed_computation_matrix_1.reshape(-1,transposed_computation_matrix_1.shape[-2],transposed_computation_matrix_1.shape[-1]) #-1 means "figure out the rest" this keeps the row and columns and basically squishes the rest of the value into one list
    final_computational_stacked_matrix_1 = np.asarray(flat_computation_matrix_1, dtype=np.complex128) #convert to complex array just for consietensiy before passing it to the solver

    #4. do checks to make sure nothing went wrong
    if not np.allclose(final_computational_stacked_matrix_1, final_computational_stacked_matrix_1.conj().transpose(0, 2, 1), atol=1e-10):
        logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")

    # 5. THE EIGENVALUES & SUMMATION 
    eigenvalues = np.linalg.eigvalsh(final_computational_stacked_matrix_1)
    total_energy = np.sum(eigenvalues[eigenvalues < 0])

    print(f"\n######################################\nSpin Down Conduction Band ↓Pattern 1↓") 
    print(f"The shape of your final matrix: {final_computational_stacked_matrix_1.shape=}")
    print(f"The values in of your final matrix: {final_computational_stacked_matrix_1=}") 
    print(f"{total_energy=}\n######################################\n") 
    # print(np.array2string(final_matrix, precision=3, separator=' + ', dtype=np.str_))

    #######################################  table 2 test  #######################################
    class_table_2 = Relation_Table(2, total_bonds = 6, basis_bonds=(2,3), basis_unit_cell = ((2,-1),(0,2)))
    
    #get your symbolic matrix
    momentum_spin_up_coupled_matrix_2 = class_table_2.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=-1)
    pre_symbolic_matrix_2 = momentum_spin_up_coupled_matrix_2(k1,k2)
    symbolic_matrix_2 = Matrix(pre_symbolic_matrix_2) #You can use Array strictly for 2-d array's instead

    #convert symbolic -> numpy computational using lamdify
    compute_matrix_2 = lambdify((k1, k2, g), symbolic_matrix_2, modules='numpy')
    #intilize the matrix with full meshgrids to create a "stack" of usable matrixs
    full_computational_matrix_2 = compute_matrix_2(k1_meshgrid, k2_meshgrid, g_val)
    
    #3. Format the matrix correctly so np.eiginvals can compute it properly
    #Make sure you moveaxis/transpose -> reshape! RESHAPING should always happen last for speed!!
    #aka transpose then flatten
    transposed_computation_matrix_2 = np.transpose(full_computational_matrix_2,axes=(2,3,0,1)) #puts all the actual values that we squished into one list at the front, this is how np.eginvalues needs it. The row and column are just basically metadata that tell np how to interpert this massive list of values
    flat_computation_matrix_2 = transposed_computation_matrix_2.reshape(-1,transposed_computation_matrix_2.shape[-2],transposed_computation_matrix_2.shape[-1]) #-1 means "figure out the rest" this keeps the row and columns and basically squishes the rest of the value into one list
    final_computational_stacked_matrix_2 = np.asarray(flat_computation_matrix_2, dtype=np.complex128) #convert to complex array just for consietensiy before passing it to the solver

    #4. do checks to make sure nothing went wrong
    if not np.allclose(final_computational_stacked_matrix_2, final_computational_stacked_matrix_2.conj().transpose(0, 2, 1), atol=1e-10):
        logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")

    # 5. THE EIGENVALUES & SUMMATION 
    eigenvalues = np.linalg.eigvalsh(final_computational_stacked_matrix_2)
    total_energy = np.sum(eigenvalues[eigenvalues < 0])

    print(f"\n######################################\nSpin Down Conduction Band ↓Pattern 2↓ Energy") 
    print(f"The shape of your final matrix: {final_computational_stacked_matrix_2.shape=}")
    print(f"The values in of your final matrix: {final_computational_stacked_matrix_2=}") 
    print(f"{total_energy=}\n######################################\n") 
    return None

def test_conduction_wrapper(
    g_value: int,
    kitaev_index: Literal[1,2,3,4,5,6,7,8,9,10,11,12,13,14], 
    coupling: Literal[-1, 0, 1, 2], 
    kx_range: Tuple[int, int] = (0, 1), 
    ky_range: Tuple[int, int] = (0, 1), 
    number_of_kx: int = 100, 
    number_of_ky: int = 100, 
    numerical: bool = True
) -> None:
    # 1. Create the 1D arrays
    kx = np.linspace(kx_range[0], kx_range[1], number_of_kx)
    ky = np.linspace(ky_range[0], ky_range[1], number_of_ky)
    
    # 2. Generate the meshgrid ONLY for the variables that sweep across a range
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # 3. Pass the resulting tuple of arrays to the main function
    test_conduction(
        g_value=g_value,
        kitaev_index=kitaev_index, 
        coupling=coupling, 
        meshgrid_tuple=(KX, KY), 
        numerical=numerical 
    )
    return None

def test_conduction(g_value: int ,kitaev_index: Literal[1,2,3,4,5,6,7,8,9,10,11,12,13,14], coupling: Literal[-1, 0, 1, 2], meshgrid_tuple: Tuple[np.ndarray, np.ndarray], numerical: bool = True):
        # I want to test any g value, any kitaev table, choosing spin, number of k values, and if it is numerical or symbolic
    """
    Tests all features of the conduction matrix.

    Args:
        coupling: Determines the type of spin coupling to test.
            -1 (spin down), 0 (no spin), 1 (spin up), or 2 (full coupling).
        numerical (bool, optional): If True, computes the matrix numerically. Defaults to True. If False just shows uncomputed rounded matricies using sympy pretty print.
    Returns:
        None
    """
    #create the symbols you will be using
    g,kx,ky = symbols('g kx ky')
    kx_value, ky_value = meshgrid_tuple

    #Basis unit cell should go into static and should be changed in Relation_Table
    phase_class = Relation_Table(kitaev_index=kitaev_index,total_bonds=6,basis_bonds=(2,3),basis_unit_cell=((2,-1),(0,2)))
    #this will run the computations just so 
    def run_computation ():
        phase_matrix = phase_class.create_momentum_transform(g_mag_field_spin_liquid_coupling_strength=g, spin_coupling=spin_coupling)
        pre_symbolic_phase_matrix = phase_matrix(kx=kx,ky=ky)
        symbolic_phase_matrix = Array(pre_symbolic_phase_matrix) #can be Matrix(pre_symbolic_matrix) if the input is a 2-d matrix, use Array if not sure

        numerial_phase_matrix = BaseMatrix.symbolic_to_numerical_matrix(symbolic_matrix=symbolic_phase_matrix, dict_of_symbols={g:g_value,kx:kx_value,ky:ky_value})

        # 5. THE EIGENVALUES & SUMMATION 
        eigenvalues = np.linalg.eigvalsh(numerial_phase_matrix)
        total_energy = np.sum(eigenvalues[eigenvalues < 0])

        # Formatting trick to cleanly print spin up vs spin down
        unicode_spin_label = "↑" if spin_coupling == 1 else "↓"
        word_spin_label = "Up" if spin_coupling == 1 else "Down"
        print(f"\n######################################")
        print(f"{unicode_spin_label}Spin {word_spin_label}{unicode_spin_label} Conduction Band Energy") 
        print(f"The shape of your final matrix: {numerial_phase_matrix.shape}")
        print(f"Total Energy: {total_energy}") 
        print(f"######################################\n")
    
    if coupling == 2:
        for i in range (2):
            spin_coupling = (-1) ** i
            run_computation()
    
    else:
        spin_coupling = coupling
        run_computation()





if __name__ == "__main__":
    # test_matrix_construction()
    # symbolic_test_coupling_up()
    # symbolic_test_coupling_down()
    # numerical_test_coupling_down()
    # numerical_test_coupling_up()
    # test_conduction_wrapper(g_value=1,kitaev_index=1,coupling=-1,numerical=True)
    test_conduction_wrapper(g_value=1,kitaev_index=2,coupling=2,numerical=True)