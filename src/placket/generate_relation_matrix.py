from typing import Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)
class Relation_Table:
    """
    Goal of this class is to take in a table with the unit cells and then 
    it will give you a unitary hermitation matrix that you can plug
    two momentum values into. 
    """
    def __init__ (self, neighbor_table:np.ndarray, total_bonds:int ,basis_bonds:tuple[int], basis_unit_cell:tuple[tuple[int]], reflection: bool = False ):
        """
        You take in 
        neighbot_table: neighbor bonds relation table, 
        total_bonds: Number of neighbor for a a single site ,
        basis_bonds: the nearest neighbor bonds you want to use as a basis, 
        basis_unit_cell: the unit cell basis vectors this *needs* to be given in terms of the nearest neighbor basis,
        reflection: if it is a symetric  for computation time a possible (not implemented yet)
        """
        if reflection and total_bonds%2 != 0 :
            raise ValueError(f"Neighbor isnt even, reflection not possible")
        elif reflection: 
            total_bonds = total_bonds/2
        #mybe something to validate dims of the tessalation pattern
        self._neighbor_table = neighbor_table
        self._total_bonds:int = total_bonds
        self._basis_bonds:tuple[int] = basis_bonds
        self._basis_unit_cell:tuple[tuple[int]] = basis_unit_cell
        if self._basis_bonds[0] > self._basis_bonds[1]:
            self._basis_bonds[1],self._basis_bonds[0] = self._basis_bonds[0],self._basis_bonds[1]
        if not all( 0 < x < self._total_bonds for x  in self._basis_bonds):
            raise ValueError(f"Given bonds are out of range, pick numbers between 1 and {self._total_bonds}")
        if len(self._basis_bonds) != 2:
            raise ValueError(f"This class only work for 2 dim systems!")
        self._steps_between_basis:int  = abs( self._basis_bonds [0] - self._basis_bonds[1])
        self._two_pi_over_N = 2 * np.pi / float(self._total_bonds)
    
    #################
    # Getter/Setter # 
    #################
    @property
    def basis_bonds():
        return self._basis_bonds
    @basis_bonds.setter
    def basis_bonds(value):
        if not isinstance(value, tuple):
            raise ValueError(f"You did not pass a tupel when setting bassis bonds")
        if len(value) != 2:
            raise ValueError(f"This class only works for 2 dim! Please make sure you tupel is of len 2")
        if any(self._total_bonds < x <= 0   for x in value):
            raise ValueError(f"You passed values above or below the accepted range is 1 to {self._total_bonds}")
        if value[0] > value[1]:
            self._basis_bonds[1],self._basis_bonds[0] = value[0],value[1]
        else: 
            self._basis_bonds = value
    
    #Private method
    def _decompose_bond(self, bond:int) -> Tuple [float,float]:
        steps_between_first_and_xbond = bond - self._basis_bonds[0]
        
        coefA_numerator = np.sin(self._two_pi_over_N * (self._steps_between_basis - steps_between_first_and_xbond))
        coefA_denom = np.sin(self._two_pi_over_N * self._steps_between_basis)
        coefA = coefA_numerator/coefA_denom
        # print(f"For the coeeficent of {self._basis_bonds[0]} is \nnumerator: {coefA_numerator} and\n denomanator: {coefA_denom}")
        coefB_numerator = np.sin(self._two_pi_over_N * steps_between_first_and_xbond)
        coefB_denom = np.sin(self._two_pi_over_N * self._steps_between_basis)
        coefB = coefB_numerator/coefB_denom
        
        return (coefA,coefB)

    #TODO: fix itterator for loop !
    #public method
    def create_momentum_transform(self):
        # ==========================================
        # 1. SETUP PHASE (Runs ONCE when called)
        # ==========================================
        unit_cell_bond_one_neighbor_basis = self._basis_unit_cell[0]
        unit_cell_bond_two_neighbor_basis = self._basis_unit_cell[1]
        
        # We can pull the inversion out of the loop entirely since it never changes!
        neighbor_basis_to_unit_cell_basis = np.array([
            [unit_cell_bond_one_neighbor_basis[0], unit_cell_bond_one_neighbor_basis[1]],
            [unit_cell_bond_two_neighbor_basis[0], unit_cell_bond_two_neighbor_basis[1]],
        ])
        inverse_neighbor_to_unit_cell = np.linalg.inv(neighbor_basis_to_unit_cell_basis)

        # Create a list to store our pre-calculated instructions
        bond_instructions = []

        for (row_idx, col_idx), value in np.ndenumerate(self._neighbor_table):
            current_neighbor = row_idx
            destination_neighbor = value
            bond = col_idx
            
            coefA, coefB = self._decompose_bond(bond)
            bond_vector_neighbor_basis = np.array([[coefA], [coefB]])
            
            new_coef = inverse_neighbor_to_unit_cell @ bond_vector_neighbor_basis
            new_coef = new_coef.flatten()
            
            # STORE the results, don't calculate the exponential yet
            bond_instructions.append({
                'row': current_neighbor,
                'col': destination_neighbor,
                'bond' : col_idx,
                'c1': new_coef[0],
                'c2': new_coef[1],
            })
        self._matrix_instructions = bond_instructions

        # ==========================================
        # 2. EXECUTION PHASE (Runs thousands of times)
        # ==========================================
        def functional_momentum_matrix(k1, k2):
            # Reset the table to zero for this specific k1, k2
            table = np.zeros((self._neighbor_table.shape[0], self._neighbor_table.shape[0]), dtype=np.complex128)
            table_debug = np.zeros((self._neighbor_table.shape[0], self._neighbor_table.shape[0], self._total_bonds), dtype=np.complex128)
            # Quickly zip through the pre-calculated instructions
            for inst in bond_instructions:
                table_debug[inst['row'], inst['col'],inst['bond']] += (k1*inst['c1'] + k2*inst['c2'])
                table[inst['row'], inst['col']] += np.exp(1j * 2 * np.pi * (k1 * inst['c1'] + k2 * inst['c2']))
                
            return table

        return functional_momentum_matrix

    
    def view_momentum_matrix(self):
            """
            Builds and returns a string representation of the momentum matrix 
            using the pre-calculated instructions.
            """
            # Ensure the instructions exist before trying to view them
            if not hasattr(self, '_matrix_instructions'):
                raise RuntimeError("You must call create_momentum_transform() before viewing the matrix.")

            N = self._neighbor_table.shape[0]
            string_table = np.full((N, N), "0", dtype=object)

            for inst in self._matrix_instructions:
                # Round to 3 decimal places for readability
                c1 = np.round(inst['c1'], 3)
                c2 = np.round(inst['c2'], 3)
                
                # Format the phase string cleanly
                terms = []
                if c1 != 0:
                    terms.append(f"{c1}*k1" if abs(c1) != 1 else ("k1" if c1 == 1 else "-k1"))
                if c2 != 0:
                    terms.append(f"{c2}*k2" if abs(c2) != 1 else ("k2" if c2 == 1 else "-k2"))
                
                # Build the exponential 
                if not terms:
                    term = "1" 
                else:
                    phase = " + ".join(terms).replace("+ -", "- ")
                    term = f"exp(2j*pi*({phase}))"
                
                # Place it in the matrix
                row, col = inst['row'], inst['col']
                if string_table[row, col] == "0":
                    string_table[row, col] = term
                else:
                    string_table[row, col] += f" + {term}"
                    
            return string_table