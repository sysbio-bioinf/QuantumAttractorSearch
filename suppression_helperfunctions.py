# Helper functions for exact deletion
from functions import *

def lastregisterindices(circ, n):
    return(list(range(circ.num_qubits))[-n:])
def forwardtransitioncircuit(rulestxt, nrTransitions=None, includeClassicalRegister=False):
    singletransitioncircuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous", includeClassicalRegister=includeClassicalRegister)
    # First generate the MultiStepCircuit:
    n = int(singletransitioncircuit.num_qubits / 2)
    if nrTransitions == None: #use transient time of n steps by default
        nrTransitions = n

    multitransitioncircuit = QuantumCircuit((nrTransitions+1)*n)
    for t in range(1, nrTransitions + 1):
        indices2compose = list(range((t - 1) * n, (t + 1) * n))  # (t-1)*n to (t+1)*n-1, 2n qubits in total to compose
        multitransitioncircuit = multitransitioncircuit.compose(singletransitioncircuit, indices2compose)

    return(multitransitioncircuit)

#Return exact tuning of phi as given by Liu and Ouyang
def optimalphi(M,N):
    #M is number of states to delete, N is size of state space
    #Check if one is in range for exact deletion M/N <= 3/4
    if (M/N > 3/4):
        Warning("Outside of M/N <= 3/4 range, using phi value for M/N=3/4 instead!")
        M = 0.75*N
    beta = np.arcsin(np.sqrt(M/N))
    phi = -2 * np.arcsin(1 / (2 * np.cos(beta)))
    return(phi)

def optimalphi2(M,N):
    #M is number of states to delete, N is size of state space
    #Check if one is in range for exact deletion M/N <= 3/4
    #if (M/N > 3/4):
    #    Warning("Outside of M/N <= 3/4 range, using phi value for M/N=3/4 instead!")
    #    M = 0.75*N
    beta = np.arcsin(np.sqrt(M/N))
    J = np.ceil(beta/(np.pi-2*beta))
    phi = -2 * np.arcsin( np.sin( np.pi / (4*J + 2) ) / (np.cos(beta)))
    return(phi, J)

#TODO Document/comment function
#TODO: If these are shifters from F+H paper, name/comment circuits accordingly as C_0^N, C_1^N etc.
"""
Implements phase shift operators as described in Fujiwara and Hasegawa paper?
Diagonal is 1 except phase shift at the last or second-to-last entry?
"""
def nqubits_lastStatesShifter(n, C, phi=np.pi / 3, circ=0, given_circ=False):
    if given_circ == False:
        shifterCirc = QuantumCircuit(n)
    else:
        shifterCirc = circ

    if C == 1:
        shifterCirc.mcp(phi, np.arange(n - 1).tolist(), n - 1)
        return (shifterCirc)

    elif C == 0:
        if n == 2:
            shifterCirc.p(phi, 1)
            shifterCirc.mcp(-phi, np.arange(0, n - 1).tolist(), n - 1)
            return (shifterCirc)

        else:
            shifterCirc.mcp(phi, np.arange(1, n - 1).tolist(), n - 1)
            shifterCirc.mcp(-phi, np.arange(0, n - 1).tolist(), n - 1)
            return (shifterCirc)  # 1,phi,1,1 diagonal -> Only phase shifts |01>

    else:
        raise ValueError("Enter -1 for phase shift on the last qubit, or -2 for the last but last.")

#TODO Document/Comment function
def nqubits_statesShifter(ind, n, phi=np.pi / 3, circ=0, given_circ=False):
    if given_circ == False:
        circ = QuantumCircuit(n)

    # Create the number representation needed for the circuit
    ind2 = np.base_repr(int(np.floor(ind / 2)))
    number_str = str(ind2)
    digits = [int(char) for char in number_str]

    for i in range(n - 1):
        try:
            if digits[-i - 1] == 0:
                circ.x(i + 1)
            elif digits[-i - 1] == 1:
                pass
        except:
            circ.x(i + 1)

    circ = nqubits_lastStatesShifter(n, ind % 2, phi=phi, circ=circ, given_circ=True)

    for i in range(n - 1):
        try:
            if digits[-i - 1] == 0:
                circ.x(i + 1)
            elif digits[-i - 1] == 1:
                pass
        except:
            circ.x(i + 1)

    return circ



