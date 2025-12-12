#makes sure the path is the root dir so we can use the proj written imports
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

#coupling range is g
#Kitaev strenght range is K
# Fermi hopping t
# Chemical_potential is ep good 
#filling for filing is fine
def main(Fermi_hopping_range: Tuple[float,float] = (0,1), NNSpin_coupling_range: Tuple[float,float]= (0,1), 
         Chemical_potential: float =0,
         Unit_cells:int = 12, Gridding:int = 10,
         Coupling:bool = True, Coupling_strength:float = 1,
         Caching:bool=True, Special_cache_func:Optional[Callable] = None,
         Logging_path:Optional[str]=None, Logging_level:Optional[str|int]=None) -> None:
    """
    What it does
    
    """
    #very first thing set up the logger
    logger = setup_logger(Log_path=Logging_path,Log_level=Logging_level)
    
    #basic energies use t = 0 for phase diagram t = 1
    
    
    ### PHYSICAL MEANING BEHIND MOST VARIBABLES USED IN THE CODE!!! ##
    if Coupling:
        t = Coupling_strength
    else:
        if Coupling_strength != 0:
            logger.warning(f"Coupling was turned off but you still passed a value to Coupling_strength:{Coupling_strength}!\nSetting coupling to 0!")
            t = 0
            g = 0
        else:
            t = 0
            g = 0
    N=Unit_cells
    points = Gridding
    ep = Chemical_potential
    #################################################################
    
    logger.debug(f"You ran the run_all.py script with these values set N = {N}, number of g,k vals {points* points}")
    # wpList = wpGenerator(MaxPatternSize)
    placketLists = kitaevList(N,N)
    logger.debug(f"Finished creating all the list of placket configurations we have {len(placketLists)} types of configurations")
    # print(f"{len (placketLists)}")
    SpecificWP = []
    for filling in [1/2,1/4,1/3]:
        finalList = [] #make a dict where you have the key being {placket,g,k:} and then the values are something tbd
        #start looping through the points
        for g in np.linspace(Fermi_hopping_range[0],Fermi_hopping_range[1],points): #orginally np.arange(0.3,0.8,0.005)
            for k in np.linspace(NNSpin_coupling_range[0], NNSpin_coupling_range[1],points):#np.arange(0.45,0.5,0.001):
                allWP = []
                #first check the cache and skip the 
                if Special_cache_func == None:
                    minIndex = check_cache("Kit",g,k,t,filling,N)
                else:
                    minIndex = check_cache("Kit",g,k,t,filling,N,path_generator=Special_cache_func)
                if minIndex:
                    #if the cache hits then skip the computation and skip the g,k value.
                    # finalList.append ([g,k,minIndex])
                    print("Skipped the wp computation it is alredy in the cache")
                    continue
                else:
                    for index,Wp in enumerate(placketLists):
                        
                        #checks that the placket configuration we are using makes sense
                        if len(Wp) == 0:
                            logger.warning(f"Check you placket lists! The {index} placket in your list is empty!\n It will be skipped so that the program does not crash but the results should not be trusted!!!")
                            print (f"The weird index is {index}")
                            continue
                        
                        #    
                        print(f"Computing for g:{g} and k:{k} and wp:{index}")
                        
                        #build each matrix seperatly with the same g,k values
                        Matrix_kit = build_kitaev_matrix(N,Wp,t,1,g,Jx=k,Jy=k,Jz=k)
                        Matrix_trig_up = build_trig_matrix(N,Wp,t,g,1,ep)
                        Matrix_trig_down = build_trig_matrix(N,Wp,t,g,-1,ep)
                        
                        #calculate the enrgies of each matrix seperatly
                        eigen_vals_kit:NDArray[np.floating] = diagonalize(Matrix_kit)
                        eigen_vals_up:NDArray[np.floating] = diagonalize(Matrix_trig_up)
                        eigen_vals_down:NDArray[np.floating] =  diagonalize(Matrix_trig_down)
                        
                        #combine the eigen vals for up and down first
                        eigen_vals_up_down = np.concatenate((eigen_vals_up, eigen_vals_down))
                        
                        #lambda expressions for gathering
                        Sum_pos:Callable = lambda lst: -0.5*np.sum(lst[lst>0])
                        Sum_lowest_n:Callable = lambda lst: sum(x for idx,x in enumerate(lst) if idx < (N*N*filling))
                        
                        #gather the energies as you please using the lambda expressions
                        summed_kit =  collect_eigen_val(eigen_vals_kit,True,Sum_pos)
                        summed_up_down =  collect_eigen_val(eigen_vals_up_down,True,Sum_lowest_n)
                        
                        print(f"Finished computing for g:{g} and k:{k} and wp:{index}")   
                        
                        #update the result
                        if type(summed_kit) == np.float64 and type(summed_up_down) == np.float64:
                            #appending 
                            allWP.append(summed_kit + summed_up_down)
                        else:
                            print("Typing went wrong!")
                            logger.critical(f"!!The typing is not right from the computation, this will crash or corrupt the cache!!\nFrom kit computation we have type {type(summed_kit)}\nFrom tirangle lattice we have type {type(summed_up_down)}")
                    print(f"done computeing all wp for g,k pair {g,k} now finding the minuimum wp!" )
                    
                    #once you do compute for all wp values then we can compute which one was the "best"
                    print(len(allWP))
                    if len(allWP) != len(placketLists):
                        print("Some plackets were siliently skipped")
                        logger.critical(f"!!Cache has been corrupted!! Not all placets were accounted for!!")
                    minEnergy = min([gse for gse in allWP ])
                    minIndex = allWP.index(minEnergy)
                    minPlackConfig = placketLists[minIndex]
                    
                    # finalList.append ([g,k,minIndex])
                    
                    #update the cache
                    if Special_cache_func == None:
                        write_cache("Kit",g,k,t,filling,N, minIndex)
                    else:
                        write_cache("Kit",g,k,t,filling,N, minIndex,Special_cache_func)
                    print(f"Appended min index {minIndex} for g,k value {g,k}! to the cache")
    #     df = pd.DataFrame(finalList, columns=["g","k", "winning_config"])
    #     df.to_csv(f"phase_results_ZoomgkpointTEST_{filling}.csv", index=False)
    #     print("Saved results to phase_results.csv ✅")
        # if finalList:
        #     df_new = pd.DataFrame(finalList, columns=["g", "k", "winning_config"])
        #     # df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        #     df_updated.to_csv(cached_result, index=False)
        #     print(f"Saved updated results to {cached_result} ✔️")
        # else:
        #     print("No new (g,k) points needed computing — CSV unchanged.")
    return None

# def find_min_plackets(N, placketLists, filling=1/2, g_range=None, K_range=None):
#     """ For each (g, K) in the given ranges, find the placket configuration index that minimizes the energy. Returns: g_vals: array of g values K_vals: array of K values winning_config: 2D array [len(g_vals), len(K_vals)] of indices """ 
#     if g_range is None: 
#         g_range = np.arange(0, 1.01, 0.1) # inclusive
#     if K_range is None: 
#         K_range = np.arange(0, 1.01, 0.1) 
#     winning_config = np.zeros((len(g_range), len(K_range)), dtype=int) 
#     for i, g in tqdm(enumerate(g_range)): 
#         for j, K in enumerate(K_range): 
#             allWP = [] 
#             for Wp in placketLists: 
#                 allWP.append( diagonalize_kitaev(N, Wp, g=g, K=K, filling=filling) ) # Find min energy and index 
#                 energies = [gse[3] for gse in allWP] 
#                 min_idx = np.argmin(energies) 
#                 winning_config[i, j] = min_idx
#                 finalList.append ([g,K,minIndex])
#     return finalList

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Kitaev–trig phase diagram computation with caching and logging."
    )
    parser.add_argument(
        "--Fermi_hopping_range",
        type=float,
        nargs=2,
        metavar=("g_min", "g_max"),
        help="Range of g values to scan."
    )
    parser.add_argument(
        "--NNSpin_coupling_range",
        type=float,
        nargs=2,
        metavar=("k_min", "k_max"),
        help="Range of k values to scan."
    )
    parser.add_argument(
        "--Chemical_potential",
        type=float,
        help="Chemical potential μ used in the trig matrices."
    )
    parser.add_argument(
        "--Unit_cells",
        type=int,
        default=12,
        help="Number of unit cells (N)."
    )
    parser.add_argument(
        "--Gridding",
        type=int,
        default=10,
        help="Number of sample points in the g and k directions."
    )
    parser.add_argument(
        "--Coupling",
        action="store_true",
        default = True,
        help="Enable coupling t = Coupling_strength. If not passed, t=0 unless Coupling_strength is 0."
    )
    parser.add_argument(
        "--Coupling_strength",
        type=float,
        default=1.0,
        help="Value of t if coupling is enabled."
    )
    parser.add_argument(
        "--Caching",
        type=lambda x: x.lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Enable or disable cache (True/False)."
    )
    parser.add_argument(
        "--Logging_path",
        type=str,
        default=None,
        help="Optional path for log file."
    )
    parser.add_argument(
        "--Logging_level",
        type=str,
        default=None,
        help="Logging level (e.g. DEBUG, INFO, WARNING)."
    )
    args = parser.parse_args()

    # Run main with unpacked arguments
    main(
        Fermi_hopping_range=tuple(args.Fermi_hopping_range),
        NNSpin_coupling_range=tuple(args.NNSpin_coupling_range),
        Chemical_potential=args.Chemical_potential,
        Unit_cells=args.Unit_cells,
        Gridding=args.Gridding,
        Coupling=args.Coupling,
        Coupling_strength=args.Coupling_strength,
        Caching=args.Caching,
        Special_cache_func= lambda calling, t, filling, Number_tiles: \
    f"{calling}NVal_{str(Number_tiles)}_TVal_{str(t)}_Filling_{str(filling).replace('/', '-')}_TEST.csv" #testing
        # Logging_path=args.Logging_path,
        # Logging_level=args.Logging_level
    )
@__cached
def build_matrix_cached():
    return build_matrix