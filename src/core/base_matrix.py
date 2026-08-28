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
                syms = sp.symbols(f'{vec_str[1:]}1:{D+1}')
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
        if not np.allclose(hermetian_matrix, hermetian_matrix.conj().transpose(0, 2, 1), atol=1e-5):
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
        
        tuple_of_symbols = tuple(dict_of_symbols.keys())

        # 1. Convert symbolic -> numpy computational using lambdify
        uninit_compute_matrix = lambdify(tuple_of_symbols, symbolic_matrix, modules='numpy')
        #intilize the matrix with full meshgrids to create a "stack" of usable matrixs
        unformated_stacked_computational_matrix = uninit_compute_matrix(*dict_of_symbols.values())
        ndims = unformated_stacked_computational_matrix.ndim
        if number_of_dim != None and number_of_dim != ndims:
            logger.warning(f"\nThe number of dim's you provided is not matching the number of dims found!\nGiven dims: {number_of_dim}\nActual dims:{ndims}\n")
        dynamic_axis = list(range(2,ndims)) + [0,1]

        # 2. Format the matrix correctly so np.eiginvals can compute it properly 
        # Make sure you moveaxis/transpose -> reshape! RESHAPING should always happen last for speed!!
        # aka transpose then flatten
        transposed_stacked_computation_matrix = np.transpose(unformated_stacked_computational_matrix,axes=dynamic_axis) #puts all the actual values that we squished into one list at the front, this is how np.eginvalues needs it. The row and column are just basically metadata that tell np how to interpert this massive list of values
        flat_stacked_computation_matrix = transposed_stacked_computation_matrix.reshape(-1,transposed_stacked_computation_matrix.shape[-2],transposed_stacked_computation_matrix.shape[-1]) #-1 means "figure out the rest" this keeps the row and columns and basically squishes the rest of the value into one list
        imaginary_flat_stacked_computational_matrix = np.asarray(flat_stacked_computation_matrix, dtype=np.complex128) #convert to complex array just for consietensiy before passing it to the solver

        # 4. do checks to make sure nothing went wrong
        BaseMatrix.check_hermitian(imaginary_flat_stacked_computational_matrix) 

        # 5. return the fully processed matrix to be computated 
        return imaginary_flat_stacked_computational_matrix 

    @staticmethod 
    def sum_neg_eignvalues (matrix):
        return [np.linalg.eignvals(matrix) < 0].sum()