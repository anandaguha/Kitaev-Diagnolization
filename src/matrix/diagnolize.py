import numpy as np
from typing import List, Any, Optional
from numpy.typing import NDArray

def diagonalize(matrix:NDArray[np.floating], Is_hermitian:bool = True) -> NDArray[np.floating]:

    Mhermitian = np.conj(matrix.T)
    assert (np.all(np.isclose(matrix,Mhermitian)))
    
    return np.linalg.eigvalsh(matrix)
    # Diagonalize M (antisymmetric, so eigenvalues are pure imaginary or zero)
    # if **kwargs:
    #     eigenvalues = np.linalg.eigvalsh(matrix)
    #     return eigenvalues

    # Sort eigenvalues by their real part
#     eigenvalues = np.sort(eigenvalues)[::-1]
#     eigenvalues = np.sort(eigenvalues)[::-1]
#     eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Compute ground state energy: sum of positive eigenvalues / 2
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    upEigen = eigenvaluesUp[eigenvaluesUp < 0]
    downEigen = eigenvaluesDown[eigenvaluesDown < 0]
    upDownEigen = np.append(eigenvaluesDown , eigenvaluesUp)
    #np.partition(upDownEigen, (N*N)*filling )[(N*N)*filling]
    sortedUpDown = np.sort(upDownEigen)
    
#     ground_state_energyUp = np.sum(upEigen)
#     ground_state_energyDown = np.sum(downEigen)
    ground_state_upDownEnergy = np.sum(sortedUpDown[:int(N*N*filling)]) 
    ground_state_energy = -0.5 * np.sum(positive_eigenvalues)
    
    #return [eigenvalues, ground_state_energy, ground_state_energyDown,ground_state_energyUp, ground_state_energy+ground_state_energyDown+ground_state_energyUp]
    return [eigenvalues, ground_state_upDownEnergy, ground_state_energy, ground_state_energy+ground_state_upDownEnergy]
# def collect_eigenvalues(Eigen_vals:List[float], filling:Optional[float], ) -> List[float]:
    
    