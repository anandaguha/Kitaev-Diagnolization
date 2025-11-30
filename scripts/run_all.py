#proj written imports
from src.matrix.generate_kitaev import build_kitaev_matrix, generateLists
from src.matrix.diagnolize import diagonalize_kitaev
from src.placket.generate_kitaev import kitaevList
from utils.setup_logger import setup_logger

#outsourced pckgs
import os
import numpy as np
import pandas as pd
import tqdm

#partial imports
from fractions import Fraction
from typing import Callable, Any, List, Tuple, Optional



def main(Fermi_hopping_range: Optional[Tuple[float,float]] = (0,1), NNSpin_coupling_range: Optional[Tuple[float,float]]= (0,1), 
         Unit_cells:int = 12, Gridding:int = 10,
         Coupling:bool = True, Coupling_strength:float = 1,
         Caching:bool=True, Special_cache:str = "",
         Logging_path:Optional[str]=None, Logging_level:Optional[str|int]=None) -> None:
    """
    What it does
    
    """
    #very first thing set up the logger
    if Logging_path == None:
        logger = setup_logger(Logging_level=Logging_level)
        logger.info(f"[Entry]: Started Loggger with default path")
    else:
        logger = setup_logger(Logging_path=Logging_path)
        logger.info(f"[Entry]: Startted Logger with user def path:f{Logging_path}")
    #basic energies use t = 0 for phase diagram t = 1
    if Coupling:
        t = Coupling_strength
    else:
        if Coupling_strength != 0:
            logger.info(f"[Entry] : Coupling was turned off but you still passed a value to Coupling_strength:{Coupling_strength}!\nSetting coupling to 0!")
            t = 0
        else:
            t = 0
    N=Unit_cells
    points = Gridding
    
    # wpList = wpGenerator(MaxPatternSize)
    placketLists = kitaevList(N,N)
    # print(f"{len (placketLists)}")
    SpecificWP = []
    for filling_str in ["1/2","1/4","1/3"]:
        finalList = [] #make a dict where you have the key being {placket,g,k:} and then the values are something tbd
        #start looping through the points
        for g in np.linspace(Fermi_hopping_range[0],Fermi_hopping_range[1],points): #orginally np.arange(0.3,0.8,0.005)
            for k in [NNSpin_coupling_range[0], NNSpin_coupling_range[1],points]:#np.arange(0.45,0.5,0.001):
                allWP = []
                for index,Wp in enumerate(placketLists):
                    if len(Wp) == 0:
                        print (f"The weird index is {index}")
                    matrix = build_kitaev_matrix(N,Wp,t,1,g,Jx=k,Jy=k,Jz=k)
                    allWP.append(diagonalize_kitaev(matrix))
                    print(f"computing for g:{g} and k:{k} and wp:{index}")
                    
                print(f"done computeing all wp for g,k pair {g,k} now finding the minuimum wp!" )
                minEnergy = min([gse[3] for gse in allWP ])
                minIndex = [List[3] for List in allWP].index(minEnergy)
                minPlackConfig = placketLists[minIndex]
                finalList.append ([g,k,minIndex])
                print(f"Appended min index {minIndex} for g,k value {g,k}!")
    #     df = pd.DataFrame(finalList, columns=["g","k", "winning_config"])
    #     df.to_csv(f"phase_results_ZoomgkpointTEST_{filling}.csv", index=False)
    #     print("Saved results to phase_results.csv ✅")
        if finalList:
            df_new = pd.DataFrame(finalList, columns=["g", "k", "winning_config"])
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            df_updated.to_csv(cached_result, index=False)
            print(f"Saved updated results to {cached_result} ✔️")
        else:
            print("No new (g,k) points needed computing — CSV unchanged.")
    return None

def find_min_plackets(N, placketLists, filling=1/2, g_range=None, K_range=None):
    """ For each (g, K) in the given ranges, find the placket configuration index that minimizes the energy. Returns: g_vals: array of g values K_vals: array of K values winning_config: 2D array [len(g_vals), len(K_vals)] of indices """ 
    if g_range is None: 
        g_range = np.arange(0, 1.01, 0.1) # inclusive
    if K_range is None: 
        K_range = np.arange(0, 1.01, 0.1) 
    winning_config = np.zeros((len(g_range), len(K_range)), dtype=int) 
    for i, g in tqdm(enumerate(g_range)): 
        for j, K in enumerate(K_range): 
            allWP = [] 
            for Wp in placketLists: 
                allWP.append( diagonalize_kitaev(N, Wp, g=g, K=K, filling=filling) ) # Find min energy and index 
                energies = [gse[3] for gse in allWP] 
                min_idx = np.argmin(energies) 
                winning_config[i, j] = min_idx
                finalList.append ([g,K,minIndex])
    return finalList

if __name__ == "__main__":
    main()