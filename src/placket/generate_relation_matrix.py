from typing import Dict
import numpy as np
Class relation_table:
    def __intit__ (self, minimum_tessalation:int, neighbors:int, reflection: bool = False ):
        if reflection:
            if neighbors%2 != 0:
                raise ValueError(f"Neighbor isnt even, reflection not possible")
            neighbors = neighbors/2
        rows = minimum_tessalation
        cols = neighbors
        self._rows = rows
        self._cols = cols
        self._table = np.zeros((rows,cols))
    
    #Public method
    def update_table(self, tessalation_number: int, bond: int, result: int) -> None:
        row = tessalation_number - 1
        col = bond - 1
        self.table [row,col] = result
        return None
        
            
            
        
        
    
        