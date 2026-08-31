import logging
import sympy as sp
logger = logging.Logger(__name__)
from typing import Dict, Any, Optional, Literal
from sympy import symbols, lambdify, nsimplify, Matrix, Array
import numpy as np
class BaseMatrix():
    """
    Should be all shared methods for any matrix calss built in thie project
    """

    def __init__ ():
        return None

    @staticmethod
    def update_lib():
        print("The letter of the update is f")
        return None

    @staticmethod
    def __process_vector(D,vec,symbolic):
        if isinstance(vec, (str, sp.Expr)):
            vec_str = str(vec)
            if vec_str.startswith("-"):            # If the input is a string, generate D symbols (1-indexed: e.g., p1, p2)
                syms = sp.symbols(f'{vec_str[1:]}1:{D+1}', real=True)
                return [-s for s in syms]
            else:
                return  sp.symbols(f'{vec_str}1:{D+1}')
        else:
            # If it's an array/list, convert elements to exact SymPy representations
            if symbolic:
                return [sp.nsimplify(val, constants=[sp.sqrt(3)], tolerance=1e-10) for val in vec]
            return np.asarray(vec)

    
    @staticmethod
    def check_hermitian (hermetian_matrix):
        if not np.allclose(hermetian_matrix, np.moveaxis(hermetian_matrix, [-2, -1], [-1, -2]).conj(), atol=1e-5): 
            logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")
        return None

    @staticmethod
    def dot_prod(D, sym1, sym2, symbolic=True):
        vec1 = BaseMatrix.__process_vector(D,sym1,symbolic)
        vec2 = BaseMatrix.__process_vector(D,sym2,symbolic)

        if symbolic:
            # SymPy naturally builds the symbolic expression when using Python's built-in sum()
            return 2 * sp.pi * sp.I * sum(v1_i * v2_i for v1_i, v2_i in zip(vec1, vec2))
        else:
            # Standard numerical dot product
            return np.dot(np.squeeze(vec1), np.squeeze(vec2)) * 2 * np.pi * 1j

    @staticmethod
    def convert_basis (original_vector_set, new_basis, stacked_by: Literal["row", "col"], symbolic:Optional[bool] = False):
        #set up everything correctly
        original_vector_set = np.asarray(original_vector_set)
        new_basis = np.asarray(new_basis)

        #DO ALL YOUR SAFTEY CHECKS!
        # Safety Check 1: Basis must be a square matrix
        if new_basis.ndim != 2 or new_basis.shape[0] != new_basis.shape[1]:
            raise ValueError(f"new_basis must be a square 2D matrix, got shape {new_basis.shape}")

        D = new_basis.shape[0]
        # Safety Check 2: Dimensions must match based on orientation
        if stacked_by == "row":
            set_cols = original_vector_set.shape[1]
            if set_cols != D:
                raise ValueError(
                    f"Row orientation mismatch: original_vector_set columns ({set_cols}) "
                    f"must equal new_basis rows ({D})."
                )
                
        elif stacked_by == "col":
            set_rows = original_vector_set.shape[0]
            if set_rows != D:
                raise ValueError(
                    f"Col orientation mismatch: new_basis dimensions ({D}) "
                    f"must equal original_vector_set rows ({set_rows})."
                )
        else:
            raise ValueError("orientation must be exactly 'row' or 'col'")


        #Do the computations in sympy or numpy based on flag
        if symbolic:
            # 1. Convert to SymPy
            sym_original = sp.Matrix(original_vector_set)
            sym_basis = sp.Matrix(new_basis)
            
            # 2. "Snap" floating point numbers to exact fractions/roots for readability
            sym_original = sp.nsimplify(sym_original, constants=[sp.sqrt(3)/2], tolerance=1e-5)
            sym_basis = sp.nsimplify(sym_basis, constants=[sp.sqrt(3)/2], tolerance=1e-5)
            
            # 3. Do exact algebraic inversion and multiplication
            change_of_basis_matrix = sym_basis.inv()
            
            if stacked_by == "row":
                new_vector_set = sym_original @ change_of_basis_matrix
            elif stacked_by == "col":
                new_vector_set = change_of_basis_matrix @ sym_original
                
            # 4. Return the sympy rep 
            return new_vector_set
        else:
            change_of_basis_matrix = np.linalg.inv(new_basis)
            if stacked_by == "row":
                new_vector_set = original_vector_set @ change_of_basis_matrix

            if stacked_by == "col":
                new_vector_set = change_of_basis_matrix @ original_vector_set

            return new_vector_set
    
    @staticmethod
    def symbolic_to_numerical_matrix(symbolic_matrix, dict_of_symbols: Dict[Any, Any], number_of_dim:Optional[int] = None) -> np.ndarray:
        
        #0. Get all the constants and do your checks
        tuple_of_symbols = tuple(dict_of_symbols.keys())
        og_shape = symbolic_matrix.shape[0]
        meshgrid_shape = list(dict_of_symbols.values())[0].shape #gives you the full shape
        if number_of_dim != None and number_of_dim != len(meshgrid_shape):
            logger.warning(f"\nThe number of dim's you provided is not matching the number of dims found!\nGiven dims: {number_of_dim}\nActual dims:{len(meshgrid_shape)}\n")
        
        # 1. Convert symbolic -> FLAT numpy computational using lambdify
        uninit_compute_matrix = lambdify(tuple_of_symbols, tuple(symbolic_matrix), modules='numpy')
        # 2. Input your mesgrid to expand out the terms with the INPUT Symbols
        uninit_compute_matrix_expanded = uninit_compute_matrix(*dict_of_symbols.values())
        # 3. Broadcast the largest array to all values so they also become that size array (turns integers into a matrix of just that number)
        uninit_compute_matrix_expanded_broadcasted = np.broadcast_arrays(*uninit_compute_matrix_expanded)
        # 4. Reshape the matrix now that that everything is the same size
        stacked_numerical_matrix = np.reshape(uninit_compute_matrix_expanded_broadcasted, (og_shape,og_shape,*meshgrid_shape))
        # 5. Transpose since your current shape is (Site,Site,Number of inputs, Number of inputs,...,Number of inputs) and the Number of inputs needs to be in the front for np.linalg.eigen
        stacked_numerical_matrix_transposed = np.moveaxis(stacked_numerical_matrix, [0,1] ,[-2,-1])
        # 5. Check your result
        BaseMatrix.check_hermitian(stacked_numerical_matrix_transposed)
        #6. Return your matrix 
        return stacked_numerical_matrix_transposed

    @staticmethod 
    def sum_neg_eignvalues (matrix):
        return [np.linalg.eignvals(matrix) < 0].sum()