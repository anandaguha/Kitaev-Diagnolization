#the convention that we use is left to right then up to down -> then v
def kitaevList(nx,ny):
    All_plack_kitev = [[] for i in range (15)]
    #we want to go through all the cols for each row so we pick a row ny, and then go thorugh all the cols nx
    for row in range(ny): 
        for col in range(nx):
            #rows and col's start at 0
            #1st phase
            All_plack_kitev[0].append(1) #all cells filled
            All_plack_kitev[14].append(0) #all cells empty
            
            #2nd phase
            if row%2 == 0:
                All_plack_kitev[1].append(1)
            if row%2 == 1:
                All_plack_kitev[1].append(0)
            
            #3rd phase
            if row%3 == 2: 
                All_plack_kitev[2].append(1)
            if row%3 == 1 or row%3 == 0:
                All_plack_kitev[2].append(0)
            
            #4th phase
            if row%3 == 1 or row%3 == 2:
                All_plack_kitev[3].append(1)
            if row%3 == 0:
                All_plack_kitev[3].append(0)
            
            #5th phase
            if row%3 == 0 and col%3 == 1:
                All_plack_kitev[4].append(1)
            elif row%3 == 1 and col%3 == 2: 
                All_plack_kitev[4].append(1)
            elif row%3 == 2 and col%3 == 0:
                All_plack_kitev[4].append(1)
#                 print(row,col)
            else:
                All_plack_kitev[4].append(0)
            
            #6th phase
            if row%3 == 0 and col %3 == 1:
                All_plack_kitev[5].append(0)
            elif row%3 == 1 and col%3 == 2: 
                All_plack_kitev[5].append(0)
            elif row%3 == 2 and col%3 == 0:
                All_plack_kitev[5].append(0)
            else:
                All_plack_kitev[5].append(1)
           
            #7th phase
            if row%4 == 0:
                All_plack_kitev[6].append(1)
            else:
                All_plack_kitev[6].append(0)
            
            #8th phase
            if row% 4 == 0 or row%4 == 3:
                All_plack_kitev[7].append(1)
            else: 
                All_plack_kitev[7].append(0)
            
            #9th phase
            if row%4 == 1: 
                All_plack_kitev[8].append(0)
            else:
                All_plack_kitev[8].append(1)
            
            #10th phase 
            if row%2 == 0: 
                All_plack_kitev[9].append(0)
            else:
                if col %2 == 0:
                    All_plack_kitev[9].append(0)
                else:
                    All_plack_kitev[9].append(1)
                      
            #11th phase
            if row%2 == 0:
                All_plack_kitev[10].append(1)
            else: 
                if col%2 == 0:
                    All_plack_kitev[10].append(1)
                else:
                    All_plack_kitev[10].append(0)
            
            #12th phase 
            if row%4 == 1 or row%4 == 3:
                All_plack_kitev[11].append(0)
            elif row%4 == 0:
                if col%2 ==1:
                    All_plack_kitev[11].append(0)
                else:
                    All_plack_kitev[11].append(1)
            else:
                if col%2 == 0: 
                    All_plack_kitev[11].append(0)
                else:
                    All_plack_kitev[11].append(1)
            
            #13th phase
            if row%4 == 0 or row%4 == 3:
                if col % 2 == 0:
                    All_plack_kitev[12].append(0)
                else:
                    All_plack_kitev[12].append(1)
            elif row%4 == 1 or row%4 == 2:
                if col %2 == 0:
                    All_plack_kitev[12].append(1)
                else:
                    All_plack_kitev[12].append(0)
            
            #14th phase
            if row%4 == 1 or row%4 == 3:
                All_plack_kitev[13].append(1)
            elif row%4 == 0:
                if col%2 ==1:
                    All_plack_kitev[13].append(1)
                else:
                    All_plack_kitev[13].append(0)
            else:
                if col%2 == 0: 
                    All_plack_kitev[13].append(1)
                else:
                    All_plack_kitev[13].append(0)
    return All_plack_kitev