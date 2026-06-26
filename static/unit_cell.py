import numpy as np
from types import SimpleNamespace
"""
We will be using this as the dominant basis thorught all parts of our code
"""
alpha_basis_1 = 2
alpha_basis_2 = 3

"""
These are the unit cells for the conduction electrons and they are based on the 
placket configuration. Conduction electrons live in the center of the hexagons
and are connected directly thourght the alpha basis.
Explanation of how this works is under the section 2.1.2 Mathimatical Methods 
https://www.overleaf.com/read/ywffgcgjgkrv#8d2d8b
"""
conduction_electron_cell = SimpleNamespace(
    table0 = np.array([
        [0,0,0,0,0,0],
    ]),
    table1 = np.array([
        [1,1,0,1,1,0],
        [0,0,1,0,0,1]
    ]),
    table2 = np.array([
        [2,2,0,1,1,0],
        [0,0,1,2,2,1],
        [1,1,2,0,0,2],
    ]),
    table3 = np.array([
        [1,2,1,2,1,2],
        [2,0,2,0,2,0],
        [0,1,0,1,0,1]
    ]),
    table4 = np.array([
        [3,3,0,1,1,0],
        [0,0,1,2,2,1],
        [1,1,2,3,3,2],
        [2,2,3,0,0,3]
    ]),
    table5 = np.array([
        [3,2,1,3,2,1],
        [2,3,0,2,3,0],
        [1,0,3,1,0,3],
        [0,1,2,0,1,2]
    ]),
    table6 = np.array([
        [3,2,1,2,3,1],
        [2,3,0,3,2,0],
        [0,1,3,1,0,3],
        [1,0,2,0,1,2]
    ]),
)
#the translation vectors for unit cells above
#GIVEN IN THE BASIS AT THE TOP
conduction_cell_vectors = SimpleNamespace(
    table0 = ((1,0), (0,1)),
    table1 = ((2,-1), (0,1)),
    table2 = ((3,0), (0,1)),
    table3 = ((2,-1), (1,1)),
    table4 = ((4,-2), (0,1)),
    table5 = ((2,0), (0,2)),
    table6 = ((2,-1), (0,2)),
)
#tells you what table in code connects to the tables in kitaev (human readability)
#https://arxiv.org/pdf/cond-mat/0506438 page 46 
conduction_table_to_kitaev = {
    "One": "table0",
    "Two": "table1",
    
    "Three": "table2",
    "Four": "table2",
    
    "Five":"table3",
    "Six":"table3",
    
    "Seven":"table4",
    "Eight":"table4",
    "Nine":"table4",
    
    "Ten":"table5",
    "Eleven":"table5",
    
    "Twelve":"table6",
    "Thirteen":"table6",
    "Fourteen":"table6",
    
}


"""
This section has to do with the coupling term in the Hamiltonian 
with the Wp values.

There are a diffrent configuration of unit cells for the same
patterns becuase these electrons (canonically called f-electrons)
are glued down to the corners of the hexagons which forms a hexagnoal lattice
unlike the conduction electrons which are glued
in the center of the hexagons and form a triangular lattice 
"""
coupling_cell = SimpleNamespace(
    table0 = np.array([
        []
    ])
)