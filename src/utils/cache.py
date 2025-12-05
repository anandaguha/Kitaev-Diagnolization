import csv
import os
import pandas as pd
import numpy as np
from functools import wraps
from typing import Callable, Any

#global_var
REQUIRED_COLUMNS = ["g", "k", "winning_config"]

#helper function
def __file_path_generator_kitaev__( calling:str, t:float, filling:float, Number_tiles:int) -> str:
        t_str,filling_str,Number_tiles_str = str(t) ,str(filling), str(Number_tiles) 
        if "/" in filling_str:
            filling_str = filling_str.replace("/","-")
        return f"{calling}NVal_{Number_tiles_str}_TVal_{t_str}_Filling_{filling_str}.csv"

# decorater cache just in case we want to also cache matrix calculations later 
# so that even though want other things than config on we could
# cache all the diagnolazations and the matrices and then 
# find diffrent values like worst config or maybe diffrent fillings?
def csv_cache(path_generator: Callable[..., Any]) -> Callable:
    """ Decorator function so you can easily cache any calculation
    you are doing using @csv_cache(func) above the function. This function
    also takes in a file name generator function as an input.
    So make sure to pass that correctly, it should be a func that
    takes the keywords of the function that is doing the calculation
    and then returning a file name.
    E.x.: @csv_cach(path_generator= lambda args: f"{args["g"]}.csv )
    """
    header = ["g", "k", "winning_config"]
    def decorator(func) -> Callable:
        @wraps(func)
        def wrapper(*args,**kwargs) -> Any:
            
            #create the file path based on the arguments we are passing to the fucntion
            File_path = path_generator(*args,**kwargs)
            #find its dir name
            Dir_name = os.path.dirname(File_path)
            #if there is a dir name make sure it exits, if it doesnt create it
            if Dir_name:
                os.makedirs(Dir_name, exist_ok=True)

            #make the csv file if it does no exist
            if not os.path.isfile(File_path):
                with open(File_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["g","k","winning_config"])
                    
            #now that we are sure that the file exists we can open it as a df
            Csv_cache =  pd.read_csv(File_path)
            #check that we are using the right column names
            if "g" not in Csv_cache.columns or "k"  not in Csv_cache.columns or "winning_config" not in Csv_cache.columns:
                raise NameError(f"The schema you passed is not correct!\nFile should  have the schema of g, k, winning config as columnn names")
            #now that we know the column names and the file exists we can check types and force them to match
            Csv_cache["g"] = pd.to_numeric(Csv_cache["g"], errors="coerce")
            Csv_cache["k"] = pd.to_numeric(Csv_cache["k"], errors="coerce")
            Csv_cache["winning_config"] = pd.to_numeric(Csv_cache["winning_config"], errors="coerce")
            
            #now that we have all our checks we can check the file for the value
            mask = (np.isclose(Csv_cache["g"], kwargs["g"]) & np.isclose(Csv_cache["k"], kwargs["k"]))
            
            #return value if we found a match
            if mask.any():
                print(f"""This value is alredy stored! No need to run computation again""")
                return Csv_cache.loc[mask,"winning_config"].iloc[0]
            #finally calculate the function if false
            else:
                print(f"""Running the computation for g,k pair {kwargs["g"]},{kwargs["k"]}""")
                result = func(*args,**kwargs)
                new_row = pd.DataFrame([{"g": kwargs["g"], "k": kwargs["k"], "winning_config": result}])
                
                #convert all the rows to numeric (just in case there are other rows maybe we wont do it in a loop?)
                new_row["g"] = pd.to_numeric(new_row["g"],errors='coerce')
                new_row["k"] = pd.to_numeric(new_row["k"],errors='coerce')
                new_row["winning_config"] = pd.to_numeric(new_row["winning_config"],errors='coerce')
                
                #one liner
                # for new_row = pd.DataFrame([pd.to_numeric(col,errors='coerce' for col in new_row])
                
                new_row.to_csv(File_path, mode='a', header=False, index=False)
                return result
        return wrapper
    return decorator

#the functional version of the cahce that can be called normally that just checks the cache
def check_cache( calling: str, 
                g: float, k: float, t:float, 
                filling:float, Number_tiles:int, 
                path_generator: Callable = __file_path_generator_kitaev__ ) -> float | None:
    """
    Normal callable function to cache (g, k) → winning_config into a CSV.

    Returns:
        The cached value if (g, k) already exists,
        or the new winning_config if it was just written.
    """
    path = path_generator(calling = calling, t = t , filling = filling, Number_tiles = Number_tiles)
    # Ensure parent directory exists
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # If file doesn't exist, create it with header
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)

    # Load CSV
    df = pd.read_csv(path)

    # Validate schema
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"CSV schema mismatch.\nExpected {REQUIRED_COLUMNS}\nFound {list(df.columns)}"
        )

    # Normalize numeric types
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build match mask
    mask = np.isclose(df["g"], g) & np.isclose(df["k"], k)

    # Cache hit
    if mask.any():
        return float(df.loc[mask, "winning_config"].iloc[0])
    else:
        # Cache miss → return nothing so the code can conintue with doing all complicated operations
        return None

#function that writes to the cache
def write_cache (calling: str, 
                 g: float, k:float, t:float, 
                 filling: float, Number_tiles:int, 
                 Winning_Config: int,
                 path_generator: Callable = __file_path_generator_kitaev__) -> None:
    
    #create the path name
    path = path_generator(calling = calling, t = t , filling = filling, Number_tiles = Number_tiles)
    # Ensure parent directory exists
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        

    # If file doesn't exist, create it with header
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)

    # Load CSV
    df = pd.read_csv(path)

    # Validate schema
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"CSV schema mismatch.\nExpected {REQUIRED_COLUMNS}\nFound {list(df.columns)}"
        )

    # Normalize numeric types
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build match mask
    mask = np.isclose(df["g"], g) & np.isclose(df["k"], k)

    # Cache hit
    if mask.any():
        print("Warning!!")
    else:
    # Cache miss → append new row
        new_row = pd.DataFrame(
            [{"g": g, "k": k, "winning_config": Winning_Config}]
        )
        print(f"""g: {g}, k: {k}, winning_config: {Winning_Config}\nNew row is\n {new_row}""")
        new_row.to_csv(path, mode="a", header=False, index=False)

    return None

#TODO
def clear_cache (calling: str, 
                 g: float, k:float, t:float, 
                 filling: float, Number_tiles:int, 
                 Winning_Config: int,
                 path_generator: Callable = __file_path_generator_kitaev__) -> None:
    ## Function that deletes the whole cache be very careful! 
    return None