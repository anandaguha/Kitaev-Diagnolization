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
#tells you if a site has a vison [site 0, site 1, ..., site n]
#1 if no vison -1 if has a vison
conduction_electron_cell_vison_configuration = SimpleNamespace(
    vison_table0 = np.array([1]),
    vison_table1 = np.array([-1,1]),
    vison_table2 = np.array([-1,1,1]),
    vison_table3 = np.array([-1,1,-1]),
    vison_table4 = np.array([-1,1,1]),
    vison_table5 = np.array([-1,-1,1]),
    vison_table6 = np.array([-1,1,1,1]),
    vison_table7 = np.array([-1,1,1,-1]),
    vison_table8 = np.array([-1,1,-1,-1]),
    vison_table9 = np.array([-1,1,1,1]),
    vison_table10 = np.array([1,-1,-1,-1]),
    vison_table11 = np.array([-1,1,1,1]),
    vison_table12 = np.array([-1,1,-1,1]),
    vison_table13 = np.array([1,-1,-1,-1]),
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

kitaev_to_conduction_table = {
    1: "table0",
    2: "table1",
    
    3: "table2",
    4: "table2",
    
    5: "table3",
    6: "table3",
    
    7: "table4",
    8: "table4",
    9: "table4",
    
    10: "table5",
    11: "table5",
    
    12: "table6",
    13: "table6",
    14: "table6",
    
}

kitaev_to_conduction_table_vison = {
    1: "vison_table0",
    2: "vison_table1",
    
    3: "vison_table2",
    4: "vison_table3",
    
    5: "vison_table4",
    6: "vison_table5",
    
    7: "vison_table6",
    8: "vison_table7",
    9: "vison_table8",
    
    10: "vison_table9",
    11: "vison_table10",
    
    12: "vison_table11",
    13: "vison_table12",
    14: "vison_table13",  
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
"""
Use this code for creating the tables maybe? 
for m in range(N):
        for n in range(N):
            # Site indices for A and B in unit cell (m, n) where m is the row index and n is the collum index
            idx_A = (m * N + n) * 2  # Sublattice A
            idx_B = idx_A + 1        # Sublattice B

            # x-bond: A(m,n) to B(m,n) IF INSWET
            if idx_A  in xlist: 
                M_hex[idx_A, idx_B] = -1*-2*Jx*1j
                M_hex[idx_B, idx_A] = -1*2*Jx*1j  
            else: 
                M_hex[idx_A, idx_B] = -2*Jx*1j
                M_hex[idx_B, idx_A] = 2*Jx*1j

            # y-bond: B(m,n) to A(m, n+1)
            n_next = (n + 1) % N
            idx_A_next = (m * N + n_next) * 2
            M_hex[idx_B, idx_A_next] = 2*Jy*1j
            M_hex[idx_A_next, idx_B] = -2*Jy*1j

            # z-bond: B(m,n) to A(m+1, n) IF INSWERT
            m_next = (m - 1) % N
            n_next = (n+1) %N 
            idx_A_next = (m_next * N + n_next) * 2
            if idx_A_next in zlist: 
                M_hex[idx_B, idx_A_next] = -1*2*Jz*1j
                M_hex[idx_A_next, idx_B] = -1*-2*Jz*1j
            else: 
                M_hex[idx_B, idx_A_next] = 2*Jz*1j
                M_hex[idx_A_next, idx_B] = -2*Jz*1j
                
"""