import time
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


start = time.time()

def cost(seq,state):

    l=len(seq) # length of the sequence

    dt=np.pi/l  # time step

    ######## pauli matrix
    sx=np.mat([[0,1],\
                 [1,0]], dtype=complex)
    sz=np.mat([[1,0],\
                 [0,-1]], dtype=complex)

    U = np.matrix(np.identity(2, dtype=complex)) #initial Evolution operator

    J=4                                   # control field strength
    #J=2

    for ii in seq:
        H =ii * J * sz + 1*sx # Hamiltonian
        U = expm(-1j * H * dt) * U  # Evolution operator

    p0=np.mat([[1],[0]], dtype=complex) #initial state
    pt=U * p0              #final state

    ########################## target state defined by yourself
    if state==1:
        target = np.mat([[0], [1]], dtype=complex)                             # south pole
    elif state==2:
        target = np.mat([[1./np.sqrt(2)], [1./np.sqrt(2)]], dtype=complex)     # equator 1
    elif state==3:
        target = np.mat([[1./np.sqrt(2)], [1./np.sqrt(2)*1j]], dtype=complex)  # equat  or 2
    elif state==4:
        target = np.mat([[np.sin(np.pi/8)], [np.cos(np.pi/8)]], dtype=complex)
    else:
        exit('invalid final state')
    err = 1-(np.abs(pt.H * target)**2).item(0).real            #infidelity (to make it as small as possible)

    return err



#v=np.random.rand(1,3) #Random 3-D vector(not necessarily an integer)#]
#v=np.asarray(v)
delta=0.01
cost_hist = []

def gradient_descent(x, dim, learning_rate, num_iterations):
    for i in range(num_iterations):
        v=np.random.rand(dim) #40 is the length of sequence here
        xp=x+v*delta
        xm=x-v*delta
        error_derivative = (cost(xp, 1) - cost(xm, 1))/(2*delta)
        if i == num_iterations-1:
            pass
            #print("Final Error", cost(x,1)) #get the error_derivative of the final iteration
        x = x - (learning_rate) * error_derivative*v
        x = x*(x<1)*(x>0)+(x>=1)
        
        cost_hist.append(cost(xp,1))
        #print (i)
        #print (cost(x,1))
    return x#cost(x,1)

def dicr(seq,n):
    out_seq = np.zeros(len(seq))
    ref_seq = np.linspace(0,1,n)
    ii=0
    for sq in seq: 
        out_seq[ii] = ref_seq[np.argmin(abs(ref_seq-sq))]
        ii+=1
    return out_seq
#x_label = np.linspace(0,1,1000,endpoint=False)

#[0.57125997, 0.999901  , 0.99959322, 0.9928275 , 0.95255001, 0.93498853]
dim=20;
nn=[2,3,4,5,6,10,20,50,100,200,500]

fid=np.zeros(11)
    
for ii in range(100):
    x     = np.random.rand(dim)
    x_trained = gradient_descent(x, dim, 0.01, 500)
    jj=0
    for n in nn:
        x_out = dicr(x_trained,n)
        fid[jj] += 1-cost(x_out,1)   
        jj+=1

fid /= 100
print (fid)


########100run
#step=6
#[0.49091396 0.71386731 0.80253568 0.81910898 0.84136981 0.8577331 0.86256551 0.86409438 0.8643975  0.86461662 0.86440809]


#######20run
#step=10
#[0.48719628, 0.61072489, 0.64456708, 0.67196765, 0.66985752, 0.66919286, 0.67425608, 0.67481841, 0.6745747 , 0.67356859, 0.67354329]
#step=20
#[0.59604636, 0.57605011, 0.62839059, 0.62617736, 0.62764441, 0.63273367, 0.63439294, 0.63398696, 0.63541904, 0.63553086, 0.63510793]
######
end = time.time()
time_taken = end - start

print ('Runtime:', time_taken)
