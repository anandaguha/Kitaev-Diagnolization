import sys 
import os
# 1. Get the path to 'scripts'
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the path to 'Kitaev-Diagnolization' (parent of scripts)
project_root = os.path.dirname(current_dir)
# 3. Add it to the system path
sys.path.insert(0, project_root)

import argparse
from src.placket.generate_relation_matrix import Relation_Table
import static.unit_cell as consts
import numpy as np

def main():
    for idx, (name,table) in enumerate(vars(consts.conduction_electron_cell).items()):
        if idx != 1:
            continue
        # print(name,type(table))
        table_with_bonds = Relation_Table(table,6,(1,2))
        matrix = table_with_bonds.create_momentum_transform()
        for k1 in range (10):
            for k2 in range(10):
                filled_matrix = matrix(k1,k2)
                print(filled_matrix.shape)
                eigenvalues = np.linalg.eigvalsh(matrix(k1,k2))
        
#self, neighbor_tabel:np.ndarray, neighbors:int, total_bonds:int ,basis_bonds:tuple[int], reflection: bool = False ):
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kitaev–trig computation")
    parser.add_argument("--tables", type=int, default=-1, help="Number of tabels ")
    main()