import numpy as np
def build_kitaev_matrix(N, wp, t, ep, g, Jx=1.0, Jy=1.0, Jz=1.0):
    # Total number of sites (2 per unit cell)
    num_sites = 2 * N * N

    #the placket list should always mod 2 == 0
    assert np.sum(wp)%2 == 0
    
    # Initialize the M matrixes
    M_hex = 1j*np.zeros((num_sites, num_sites))
    
    zlist, xlist = generateLists(wp,N,N)
    #print(xlist)
    #print(zlist)
    
    # Loop over unit cells for hex matrix
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
    return M_hex

def build_trig_matrix(N:int,wp,t:float,g:float,Up_down:int, ep:float,Jx:float=1.0,Jy:float=1.0,Jz:float=1.0)->np.ndarray[float]:
    num_sites = N * N
    # M_tup = 1j*np.zeros((num_sites, num_sites))
    # M_tdown = 1j*np.zeros((num_sites, num_sites))
    M_trig = 1j*np.zeros((num_sites, num_sites))
    
    #loop over unit cells for trig matrix            
    for m in range(N):
        for n in range(N):
            # Site indices for A and B in unit cell (m, n) where m is the row index and n is the collum index
    #             idx_A = (m * N + n) * 2  # Sublattice A
    #             idx_B = idx_A + 1        # Sublattice B
            idx = m*N + n
            idx_x = (m)*N + (n+1)%N
            idx_y = ((m+1)%N)*N + (n)
            idx_z = ((m-1)%N)*N + (n+1)%N
            #x-bond 
            
            M_trig[idx,idx_x] += -t
            M_trig[idx_x,idx] += -t
            
            # M_tup[idx,idx_x] += -t
            # M_tup[idx_x,idx] += -t
            
            # M_tdown[idx,idx_x] += -t
            # M_tdown[idx_x,idx] += -t
            
            #ybond
            M_trig[idx,idx_y] += -t
            M_trig[idx_y,idx] += -t
            
            # M_tup[idx,idx_y] += -t
            # M_tup[idx_y,idx] += -t
            
            # M_tdown[idx,idx_y] += -t
            # M_tdown[idx_y,idx] += -t
            
            #z-bond
            M_trig[idx,idx_z] += -t
            M_trig[idx_z,idx] += -t
            
            # M_tup[idx,idx_z] += -t
            # M_tup[idx_z,idx] += -t
            
            # M_tdown[idx,idx_z] += -t
            # M_tdown[idx_z,idx] += -t
            
            #self 
#             print(f"  the length of our placket is {len(wp)}, the legnth of up and down is{len(M_tup)}, {len(M_tdown)}, and the idx is {idx}")
            if len(wp)==0:
                print(wp)
            if wp[idx] == 0: 
                g = g if Up_down == 1 else -g
                M_trig[idx,idx] += g+ep
                # M_trig[idx,idx] = M_trig[idx,idx] + g + ep if Up_down == 1 else M_trig[idx,idx] - g + ep
                # M_tup[idx,idx] += g+ep
                # M_tdown[idx,idx] += -g+ep
            if wp[idx] == 1:
                g = -g if Up_down == 1 else g
                M_trig[idx,idx] += g+ep
                # M_trig[idx,idx] = M_trig[idx,idx] - g + ep if Up_down == 1 else M_trig[idx,idx] + g + ep
                # M_tup[idx,idx] += -g+ep
                # M_tdown[idx,idx] += g+ep
    return M_trig

def generateLists (plackList,nx,ny):
    PlackPairList = []
    idx1 = -1
    idx2= -1
    zlist = []
    xlist = []
    #print(len(plackList))
    for idx,plack in enumerate(plackList):
        #print(idx)
        if plack == 1 and idx1 == -1 and idx2 == -1:
            idx1 = idx
        elif plack == 1 and idx1 != -1:
            PlackPairList.append((idx1,idx))
            idx1 = -1 
    #print(plackList,PlackPairList)
    for p1,p2 in PlackPairList:
        r1= p1//ny
        r2= p2//ny
        c1= p1%ny
        c2= p2%ny
        #print(r1,c1,r2,c2)
        site1 = (r1*2*nx)+2*c1 
        site2 = (r2*2*nx)+2*c2
        for col in range(0,c2-c1):
            zlist.append(site1+2*(1+col))
        if r2-r1 > 0:
            for row in range(0,r2-r1):
                xlist.append(site2-row*2*nx)
        elif r2-r1 < 0:
            for row in range(0,r1-r2):
                xlist.append(site2+((row+1)*2*nx))

    return zlist,xlist
