import csv
import os
import pandas as pd
import numpy as np
from functools import wraps
from typing import Callable, Any

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
    def decorater(func) -> Callable:
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
    return decorater
        
def file_path_generator_kitaev( t:float, filling, Number_tiles:int) -> str:
        t,filling,Number_tiles = str(t) ,str(filling), str(Number_tiles) 
        if "/" in filling:
            filling = filling.replace("/","-")
        return f"NVal_{Number_tiles}_TVal_{t}_Filling_{filling}.csv"
        
