from typing import Dict, Tuple
import numpy as np
class Relation_Table:
    """
    Goal of this class is to take in a table with the unit cells and then 
    it will give you a unitary hermitation matrix that you can plug
    two momentum values into. 
    """
    def __init__ (self, neighbor_tabel:np.ndarray, total_bonds:int ,basis_bonds:tuple[int], basis_unit_cell:tuple[tuple[int]], reflection: bool = False ):
        """
        You take in 
        neighbot_table: neighbor bonds relation table, 
        total_bonds: Number of neighbor for a a single site ,
        basis_bonds: the nearest neighbor bonds you want to use as a basis, 
        basis_unit_cell: the unit cell basis vectors this needs to be given in terms of the nearest neighbor basis,
        reflection: if it is a symetric  for computation time a possible (not implemented yet)
        """
        if reflection and total_bonds%2 != 0 :
            raise ValueError(f"Neighbor isnt even, reflection not possible")
        elif reflection: 
            total_bonds = total_bonds/2
        #mybe something to validate dims of the tessalation pattern
        self._neighbor_table = neighbor_tabel
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
        steps_between_first_and_xbond = abs(bond - self._basis_bonds[0])
        
        coefA_numerator = np.sin(self._two_pi_over_N * (self._steps_between_basis - steps_between_first_and_xbond))
        coefA_denom = np.sin(self._two_pi_over_N * self._steps_between_basis)
        coefA = coefA_numerator/coefA_denom
        print(f"For the coeeficent of {self._basis_bonds[0]} is \nnumerator: {coefA_numerator} and\n denomanator: {coefA_denom}")
        coefB_numerator = np.sin(self._two_pi_over_N * steps_between_first_and_xbond)
        coefB_denom = np.sin(self._two_pi_over_N * self._steps_between_basis)
        coefB = coefB_numerator/coefB_denom
        
        return (coefA,coefB)

    #TODO: fix itterator for loop !
    #public method
    def create_momentum_transform (self):
        table = np.zeros ((self._total_bonds,self._total_bonds))
        unit_cell_bond_one_neighbor_basis = self._basis_unit_cell[0]
        unit_cell_bond_two_neighbor_basis = self._basis_unit_cell[1]
        def functional_momentum_matrix(k1,k2):
            for (row_idx, col_idx), value in np.ndenumerate(self._neighbor_table):
                
                current_neighbor:int = row_idx
                destination_neighbor:int = value
                bond:int = col_idx
                
                coefA,coefB = self._decompose_bond(bond)
                
                bond_vector_neighbor_basis = np.array([[coefA], [coefB]])
                neighbor_basis_to_unit_cell_basis = np.array([
                    [unit_cell_bond_one_neighbor_basis[0], unit_cell_bond_one_neighbor_basis[1]],
                    [unit_cell_bond_two_neighbor_basis[0], unit_cell_bond_two_neighbor_basis[1]],
                ])
                inverse_neighbor_to_unit_cell = np.linalg.inv(neighbor_basis_to_unit_cell_basis)
                new_coef = inverse_neighbor_to_unit_cell @ bond_vector_neighbor_basis
                new_coef = new_coef.flatten()
                
                print(f"Shape of final matrx: {new_coef.shape}\nShape of first coef: {new_coef[0].shape}\nShpae of first coef:{new_coef[1].shape}")
                table[current_neighbor, destination_neighbor] += np.exp(1j * (k1*new_coef[0] + k2*new_coef[1]))
            
            return table
        return functional_momentum_matrix


    
        