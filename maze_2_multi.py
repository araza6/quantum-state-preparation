"""

The code is written on top of tutorials given at: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
from scipy.linalg import expm

class Maze( object):
    def __init__(self,
        action_space=[0,1,2],
        dt=0.1):
        super(Maze, self).__init__()
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.state = np.array([1,0,0,0])
        self.nstep=0 ##count number of step at each episode
        self.dt=dt
    def reset(self):

        # return observation
        self.state = np.array([1,0,0,0])
        self.nstep = 0 #reset number of step at each episode

        return self.state

    def step(self, action):


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi)

        J = 4  # control field strength
        # J=2
        ######## pauli matrix
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex))  # initial Evolution operator

        H = J *float(action)/(self.n_actions-1) * sz + 1 * sx  # Hamiltonian
        U = expm(-1j * H * self.dt)  # Evolution operator


        # if action == 1:
        #     H = J * sz + 1 * sx  # Hamiltonian
        #     U = expm(-1j * H * dt)  # Evolution operator
        # elif action == 0:
        #     H = 1 * sx  # Hamiltonian
        #     U = expm(-1j * H * dt)  # Evolution operator
        # else:
        #     error('invalid action')

        psi = U * psi  # final state
        ########################## target state defined by yourself
        target = np.mat([[0], [1]], dtype=complex)  # south pole
        #target = np.mat([[1. / np.sqrt(2)], [1. / np.sqrt(2)]], dtype=complex)  # equator 1
        #target = np.mat([[1. / np.sqrt(2)], [1. / np.sqrt(2) * 1j]], dtype=complex)  # equator 2
        #target = np.mat([[np.sin(np.pi / 8)], [np.cos(np.pi / 8)]], dtype=complex)
        #print target
        err = 1 - (np.abs(psi.H * target) ** 2).item(0).real  # infidelity (to make it as small as possible)
################################################################

        #rwd =  10*(err < 10e-3)  # give reward only when the error is small enough
        rwd = 10 * (err<0.5)+100 * (err<0.1)+5000*(err < 10e-3)   #or other type of reward

        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt )  #end each episode if error is small enough, or step is larger than 2*pi/dt
        if done:
            if err < 10e-3:
                pass
                #print ("this episode find terminal sucessfully with a minor error:", err)
            else:
                pass
                #print ("Unsuccessful termination with considerable error:", err) #err = 1-fidelity
        self.nstep +=1  # step counter add one

        psi=np.array(psi)
        ppsi = psi.T
        self.state = np.array(ppsi.real.tolist()[0] + ppsi.imag.tolist()[0])

        return self.state, rwd, done, err

    def render(self):
        pass



