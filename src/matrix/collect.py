import numpy as np
from numpy.typing import NDArray
from typing import List, Callable
def collect_eigen_val(Eigen_vals:NDArray[np.floating],Sort:bool=False,process:Callable= lambda x:x) -> List[float] | float | int:
    if Sort:
        Eigen_vals_sorted = np.sort(Eigen_vals)
        Eigen_vals_processed = process(Eigen_vals)
    return Eigen_vals_processed


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