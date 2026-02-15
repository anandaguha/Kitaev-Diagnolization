import numpy as np
from types import SimpleNamespace

params = SimpleNamespace(
    table1 = np.array([
        [1,1,0],
        [0,0,1]
    ]),
    table2 = np.array([
        [2,2,0,1,1,0],
        [0,0,1,2,2,1],
        [1,1,2,0,0,2]
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
    ])
)
table_to_kitaev = {
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