import sys
from pathlib import Path

# Add project root (directory containing src/) to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



###importing from project###

###########   Matrix building, diagnolizing, and collecting   ###########    
from src.matrix.generate_kitaev import build_kitaev_matrix, build_trig_matrix
from src.matrix.diagnolize import diagonalize
from src.matrix.collect import collect_eigen_val
#########################################################################

###########  Building placket configurations   ###########    
from src.placket.generate_kitaev import kitaevList
##########################################################

###########  Building placket configurations   ###########
from src.utils.cache import check_cache, write_cache
##########################################################

###########  Importing logger   ###########
from src.utils.setup_logger import setup_logger
##########################################################

###end importing from projects###

#outsourced pckgs
import os
import numpy as np
import pandas as pd
import tqdm
import argparse

#outsourced partial imports
from fractions import Fraction
from typing import Callable, Any, List, Tuple, Optional
from numpy.typing import NDArray

#2 dim is for the g,k values 
#1 dim is for the diffrent plackets
#1 dim is for the integral, multiple phase space points

#generate g,k
#g_k_values = [np.arange(0,1,100), np.arange(0,1,100)]
#for each g,k (this k is a constant for coupling or something, not a momentum value in phase space) value follow this procedure:
#compile one phase into a stack - this is 1 integral
#stack each "integral" on top of eachother - stacked integrals
#evalute all integrals together to reduce the stack into stacked energies
#find minimum energy of the stack  


generate_stack()
integrate_stack()

find_



if __name__ == "__main__":
    calculate_Phase_plot