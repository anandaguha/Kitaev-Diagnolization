import logging
logger = logging.Logger(__name__)
from typing import Dict, Any, Optional
from sympy import symbols, lambdify, nsimplify, Matrix, Array
import numpy as np
class BaseMatrix():
    """
    Should be all shared methods for any matrix calss built in thie project
    """

    def __init__ ():
        return None
    
    @staticmethod
    def check_hermitian (hermetian_matrix):
        if not np.allclose(hermetian_matrix, hermetian_matrix.conj().transpose(0, 2, 1), atol=1e-5):
            logger.warning("Warning: Matrices are not strictly Hermitian! Imaginary leaks detected.")
        return None

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
    def calculate_eignvaluse (matrix):
        return [np.linalg.eignvals(matrix) < 0].sum()