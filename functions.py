import random
from qiskit import *
import numpy as np
import importlib
import copy
import math
#from qiskit_aer import AerSimulator
#from qiskit.providers.aer.noise import NoiseModel
#from qiskit_aer.noise import NoiseModel #old version
#from qiskit.providers.aer import Aer #Alternative import
from itertools import *
#from qiskit.test.mock import *
import warnings
#import braket.circuits
from datetime import datetime
import boto3
#from braket.aws import AwsDevice
#from braket.devices import LocalSimulator
#from braket.circuits import Circuit
import re #regex for parsing circuit to ionq
import collections
from functools import reduce
import pickle
from itertools import starmap
from operator import mul

#20.09.22: Helper function from Mirko
def checksin(x):
  if x <=1:
    return x
  else:
    return checksin(x-2)

# HELPER FUNCTIONS
"""
Normalize count results in weight dictionaries to sum to 1 and thus correspond to probabilities of measurement.
Can be sorted states from largest to smallest probability. Can round results to 'digits' decimals.
"""
def outputformat(d, target=1.0, normalizeOutput=True, sortOutput=True, digits=None):
    #round results to number of 'digits' after comma, do not round if digits=None
   d = {k: v for k, v in d.items() if v != 0} #take out elements with value==0
   raw = sum(d.values())
   factor = target/raw
   if normalizeOutput and sortOutput:
       if digits == None:
            return {key: (value * factor) for key, value in sorted(d.items(), reverse=True, key=lambda item: item[1])}
       else:
            return {key:round(value*factor,digits) for key, value in sorted(d.items(), reverse=True, key=lambda item: item[1])}
   elif normalizeOutput and not sortOutput:
       if digits == None:
            return {key: (value * factor) for key, value in d.items()}
       else:
            return {key:round(value*factor,digits) for key, value in d.items()}
   elif not normalizeOutput and sortOutput:
       return {key: value for key, value in
               sorted(d.items(), reverse=True, key=lambda item: item[1])}
   else:
       return d

"""
Convert a dictionary with state measurement probabilities to a list. Index 0 corresponds to key '000', index 1 to '001' etc. 
Always has length 2^N, even if not all states were measured in dict.
"""
def probdict2list(dict):
    N = len(list(dict.keys())[0]) #get first key in the dict -> get number of characters in its string = N
    problist = list(np.repeat(0, 2**N))
    for key in dict.keys():
        keyint = int(key, 2) #already 0-indexed and little endian, i.e. 000->0, 001->1,...,111->7
        problist[keyint] = dict[key]
    return problist

"""
Classical fidelity to compare two probability distributions over the same index set, given as F_s by Lubinski et al.
"""
def classical_fidelity(p,q):
    dicts = [p, q]
    keysUnion = list(reduce(set.union, map(set, map(dict.keys, dicts))))
    #print(keysUnion)

    for k in keysUnion:
        if k not in list(p.keys()):
            #print("Key " + str(k) + " is not in the dict p!")
            p[k] = 0
        if k not in list(q.keys()):
            #print("Key " + str(k) + " is not in the dict q!")
            q[k] = 0

    #print("Matching dicts p and q:")
    ordered_p = dict(collections.OrderedDict(sorted(p.items())))
    ordered_q = dict(collections.OrderedDict(sorted(q.items())))

    #Calculate fidelity
    fidelity = 0
    for k in keysUnion:
        fidelity += np.sqrt(ordered_p[k]*ordered_q[k])
    return fidelity**2

"""
Normalized fidelity as defined by Lubinski et al. (2021).
"""
def normalized_fidelity_Lubinski(idealDistr, outputDistr, uniformDistr):
    return (classical_fidelity(idealDistr, outputDistr) - classical_fidelity(idealDistr, uniformDistr))/(1 - classical_fidelity(idealDistr, uniformDistr))

"Averaging distributions for quantum counting"
def avgOfDistr(distr):
    #State is already normalised, need to sum up all key*val, key is nrResults of counting, val is prob.weight
    avg = sum(starmap(mul, distr.items())) / sum(distr.values())
    return(avg)

"""
Mapping of gate names between Qiskit and Amazon Braket/IonQ.
"""
#keys are qiskit gates -> vals are ionq names that they should be parsed to
#v gate in ionq is qiskits square root of not sx, vi is sxdg
gatenameMapping_Qiskit_IonQ = {"rzz":"zz", "sxdg":"vi", "rxx":"xx", "z":"z", "swap":"swap", "sx":"v",
                   "y":"y", "rx":"rx", "rz":"rz", "s":"s", "ry":"ry", "yy":"yy", "tdg":"ti",
                   "h":"h", "i":"i", "x":"x", "sdg":"si", "cx":"cnot", "t":"t"}
#ionq_basisgates_qiskitnames = ['rzz', 'sxdg', 'rxx', 'z', 'swap', 'sx', 'ry', 'rx', 'rz', 's', 'ry', 'yy', 'tdg', 'h', 'i', 'x', 'sdg', 'cx', 't']

"""
Takes a Qiskit circuit and transpiles it to the IonQ set of basis gates, given a mapping dictionary of gate names between these two frameworks.
The transpiled circuit's QASM representation is used to create a braket.circuits.Circuit() object with these exact gates, which can then be run on Braket.
"""
def QiskitQASM2IonQCompiler(QiskitCircuit, gatenameMapping):
    TranspiledQiskitCircuit = transpile(QiskitCircuit, basis_gates=list(gatenameMapping.keys()))
    IonQCircuit = braket.circuits.Circuit()
    QiskitQASM = TranspiledQiskitCircuit.qasm().split("\n") #list of operations
    #Parse its instruction into braket ie line cx q[4],q[7] -> cnot(4,7)
    totallines = len(QiskitQASM)
    for linenr in range(4,totallines-1):
        operatorstr = QiskitQASM[linenr]
        #print("operatorstr = " + operatorstr)
        braketstr = ""
        opname_qubits = operatorstr.split(" ")
        opname = opname_qubits[0]
        qubits = opname_qubits[1]
        #split operatorname on ( if it occurs
        # -> first is gate name, translate to ionq gate name,
        # second is angle -> replace pi with np.pi
        gatename_operand = opname.split("(")
        gatename = gatename_operand[0]
        operand = ""
        if len(gatename_operand) > 1:  # e.g. rz(-pi/8) q...; but not if e.g. h q...; or cx q...;
            operand = gatename_operand[1]
            operand = operand.replace("pi", "np.pi")
        qubitnumbers = re.findall(r'\d+', qubits)
        braketstr += gatenameMapping[gatename] + "("
        for q in qubitnumbers:
            braketstr += str(q)
            braketstr += ","
        braketstr += operand
        braketstr = braketstr[:-1]
        braketstr += ")"
        #print("braketstr = " + braketstr)
        exec("IonQCircuit." + braketstr)

    return IonQCircuit

"""
Function for parsing results from IonQ device back into same notation as Qiskit.
IonQ always measures all qubits, and its order/endian is opposite to Qiskit.
"""
def ionqDict2qiskitNotation(ionqDict, qubitindices, invertEndian=True, normalizeOutput=True, sortOutput=True):
    qiskitDict = {}
    for key in ionqDict.keys():
        reorderedShortKey = ""
        for qind in qubitindices:
            reorderedShortKey += key[qind]
        if reorderedShortKey not in list(qiskitDict.keys()):
            qiskitDict[reorderedShortKey] = 0
        qiskitDict[reorderedShortKey] += ionqDict[key]

    qiskitDict = outputformat(qiskitDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput)

    if invertEndian == True: #go back to qiskit endian
        invertedQiskitDict = {}
        for key in list(qiskitDict.keys()):
            invertedkey = key[::-1]
            invertedQiskitDict[invertedkey] = qiskitDict[key]
        return invertedQiskitDict
    else:
        return qiskitDict

"""
Parser that read BoolNet formatted txt file with rules and generates a new .py file including all the update functions as required in tweedledum
"""
def parse_BoolNetRules(rulestxt, saveDir=None):
    #NOTE: Required that final regulatory line in the txt rule file is ended with a linebreak, next line is empty

    splitfilename = str.split(rulestxt, sep=".")[0]
    splitfilename = str.split(splitfilename, sep="/")
    netname = splitfilename[len(splitfilename) - 1] #name of network with preceding path or .txt extension

    rulesvar = open(rulestxt, 'r')
    lines = rulesvar.readlines()
    count = 0

    genenamestring = [0 for i in range(len(lines))]
    genecount = 0
    for line in lines:
        splitline = line.split(sep=",")
        genenamestring[genecount] = splitline[0].lower() #must have lowercase gene names
        genecount += 1
    genenamestring = genenamestring[1:]

    genenamestring_intmap = ""
    # genenamestring_intmap = A: Int1, B: Int1, C: Int1
    for g in genenamestring:
        genenamestring_intmap += g
        genenamestring_intmap += ": Int1, "
    #Final two chars, comma + space not needed at end
    genenamestring_intmap = genenamestring_intmap[:-2]

    # Generate new python file
    if saveDir == None:
        newpyfilename = netname + "_rulefile.py"
    else:
        newpyfilename = saveDir + netname + "_rulefile.py"

    with open(newpyfilename, 'w+') as pyfileRules:
        pyfileRules.write("#Libraries: \n")
        pyfileRules.write("from qiskit import * \n")
        pyfileRules.write("from qiskit.circuit import classical_function, Int1 \n")
        #pyfileRules.write("from qiskit.circuit.classicalfunction import classicalfunction \n") #Changed 7.9.23, ubuntu 22.04, python 3.9, qiskit 0.44.1
        #pyfileRules.write("from qiskit.circuit.classicalfunction.types import Int1 \n") #Changed 7.9.23, ubuntu 22.04, python 3.9, qiskit 0.44.1
        pyfileRules.write(" \n")
        pyfileRules.write("#Regulatory functions to synthesize into circuits using Qiskit: \n")
        for line in lines: #read every rule in txt file
            if count == 0:
                count += 1
                continue #skip targets, factors header line
            else:
                #print(line)
                pyfileRules.write("@classical_function \n")
                pyfileRules.write("def g" + str(count-1) + "_update(" + genenamestring_intmap + ") -> Int1:\n")

                # Write line: paste("\t" + "return (" + regstring + ")
                #print("Count=" + str(count))
                BN_regstring = line.split(sep=",")[1]
                #print(BN_regstring)
                BN_regstring = BN_regstring[1:-1] #remove first space and final linebreak from rulestring

                #replace & | and ! symbols
                BN_regstring = BN_regstring.replace("&", "and")
                BN_regstring = BN_regstring.replace("|", "or")
                BN_regstring = BN_regstring.replace("!", "not ")

                pyfileRules.write("\t return (" + BN_regstring.lower() + ") \n")
                pyfileRules.write(" \n")

                count += 1

    pyfileRules.close()
    #load all single gene update functions from parsed py file
    with open(newpyfilename) as f:
        exec(compile(f.read(), newpyfilename, "exec"))


"""
Given a text file with update rules in BoolNet-format, generates a 2*n,n quantum circuit that updates the entire network by one timestep if executed.
If update="asynchronous" a random update order is generated and circuits use the updated state of previously updated nodes.
"""
def synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous", includeClassicalRegister=True, updateorder=None):
    parse_BoolNetRules(rulestxt)

    #g0_update() etc functions only become available inside this function when exec is called again explicitely
    splitfilename = str.split(rulestxt, sep=".")[0]
    splitfilename = str.split(splitfilename, sep="/")
    netname = splitfilename[len(splitfilename) - 1]
    newpyfilename = netname + "_rulefile"
    importrules = __import__(newpyfilename) #no error, does import module

    rulesvar = open(rulestxt, 'r')
    lines = rulesvar.readlines()
    n = len(lines)-1 #-1 to account for header line

    #If circuit has to be turned into a gate then it must not include a classical register
    if includeClassicalRegister:
        FullCircuit = QuantumCircuit(2 * n, n)
    else:
        FullCircuit = QuantumCircuit(2 * n)
    qr = QuantumRegister(2*n)
    cr = ClassicalRegister(n)

    CircuitList = list(np.repeat(0, n))

    if update == "synchronous":
        for g in range(n):
            exec("CircuitList[" + str(g) + "] = importrules.g" + str(g) + "_update.synth()") # individual gene's update circuit with n+1 qubits (n inputs, one output)
            if includeClassicalRegister:
                CircuitList[g].add_register(cr)
            # All individual circuit QuantumCircuit objects are synthesized, stored in list, and have cr added
            # FullCircuit = FullCircuit.compose(GenegUpdate, [0,..., n-1, n+g], [0,...,n-1]) #n+g = n,..., 2n-1
            outputqubitlist = list(range(n))
            outputqubitlist.append(n+g)
            if includeClassicalRegister:
                FullCircuit = FullCircuit.compose(CircuitList[g], outputqubitlist, list(range(n)))
            else:
                FullCircuit = FullCircuit.compose(CircuitList[g], outputqubitlist)
            #second argument=qubits of self to compose onto, last qubit is varying output of gX(t+1), third argument is classical register (unchanging)

        #DO NOT MEASURE YET
        #FullCircuit.measure(list(range(N, 2*N)), list(range(0,n)))
        return (FullCircuit)

    elif update == "asynchronous":
        if updateorder == None: #if no order is specified, generate a random order
            updateorder = list(range(n))
            random.shuffle(updateorder)
        elif len(updateorder) != n:
            print("updateorder needs to be of length n")
        print("Update order of the asynchronous circuit is " + str(updateorder))
        outputqubitlist = list(range(n+1))

        for g in range(n):
            gene2update = updateorder[g]
            exec("CircuitList[" + str(gene2update) + "] = importrules.g"+str(gene2update)+"_update.synth()") # individual gene's update circuit with n+1 qubits (n inputs, one output)
            CircuitList[gene2update].add_register(cr)
            # All individual circuit QuantumCircuit objects are synthesized, stored in list, and have cr added
            outputqubitlist[n] = n+gene2update
            FullCircuit = FullCircuit.compose(CircuitList[gene2update], outputqubitlist, list(range(n)))
            outputqubitlist[gene2update] = n+gene2update
            #second argument=qubits of self to compose onto, last qubit is varying output of gX(t+1), third argument is classical register (unchanging)

        #DO NOT MEASURE YET
        #FullCircuit.measure(list(range(n, 2*n)), list(range(0,n)))
        return (FullCircuit)

    else:
        return ("Not a valid updating scheme")


"""
Generate circuit that inits starting states with Ry gates, to be composed with single/multistep circuits + .measure() statement
"""
def initActivityCircuitGenerator(activities, Tmax=1, thetatype="angle", method="exact"):
    # map starting state gene activities=[0,1] (KO->unperturbed=0.5->OE) to theta=[0,pi]
    # use activities=np.repeat(0.5,n) for allH default init
    n = len(activities)
    initcirc = QuantumCircuit((Tmax+1) * n, n) #make size adaptable, can also be called as init for exact multistep transition
    #Map all thetas to linear interval
    if thetatype == "linear_corrected":
        #compensate for cosine in measurement - linear passed on activity directly and this worked
        activities = [2*np.arccos(np.sqrt(activity)) for activity in activities]
    #elif thetatype == "angle_corrected":
    #    #compensate for cosine in measurement
    #    activities = [2*np.arccos(activity/180) for activity in activities]
    elif thetatype == "angle":
        #Map [0,180] to [0,1]
        activities = [activity/180 for activity in activities]
    elif thetatype == "radian":
        #Map [0,pi] to [0,1]
        activities = [activity/np.pi for activity in activities]

    for g in range(len(activities)): #first n qubits are altered by Ry gates, remaining n output qubits remain in default |0> state
        #if thetatype == "linear":
            initcirc.ry(theta=activities[g]*np.pi, qubit=g)
        #    print("Theta value in linear range is " + str(activities[g]) + ", in radians it is " + str(activities[g] * np.pi) + ".")
        #elif thetatype == "angle":
        #    initcirc.ry(theta=(activities[g]/180) * np.pi, qubit=g)
        #    print("Theta value in degrees is " + str(activities[g]) + ", in radians it is " + str((activities[g]/180) * np.pi) + ".")
        #elif thetatype == "radian":
        #    initcirc.ry(theta=activities[g], qubit=g)
        #else:
        #    return("Error: Argument 'theta' should be either 'linear', 'angle' or 'radian'.")

    return(initcirc)



#SYNCHRONOUS STATE TRANSITIONS
"""
Exact multistep circuit transitions which can distinguish if Ry gates should serve only as initial expression bias or be used for true perturbations.
Also allows for addition of noise models. Performs synchronous state transitions.
"""
def allparam_exact_multiTransition_synchronous(rulestxt, Tmax=2, nrshots=100, initActivities=None, initPerturbed=None,
                                   thetatype="linear", normalizeOutput=True, sortOutput=True, backend=None,
                                   transpileCircuit=True, addNoise=True, optimization_level=0,
                                   approximation_degree=None, returnCircuitDepth=False, seed_transpiler=None, seed_simulator=None):
    if isinstance(rulestxt, QuantumCircuit):
        circuit = rulestxt #Directly provided circuit
    else:
        circuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous") #Provided path to rules -> synthesize circuit
    #First generate the MultiStepCircuit:
    n = int(circuit.num_qubits/2)
    totalqubits = (Tmax + 1) * n #Needed 2*n qubits for single transition, now (T+1)*n qubits needed for full circuit
    cr = ClassicalRegister(n)

    if initPerturbed == None:
        initPerturbed = list(np.repeat(False, n))

    if initActivities == None:
        InitCircuit = QuantumCircuit(totalqubits, n)
        #default init all-H
        for q in range(0,n):
            InitCircuit.h(q)
    else:
        InitCircuit = initActivityCircuitGenerator(activities=initActivities, Tmax=Tmax, thetatype=thetatype) #Returns only 2n circuit, not (T+1)*n, need to add (T-1)*n qubits

        #all non 0.5 init genes now had an Ry gate added, no need to expand initActivityCircuitGenerator
        #for non-perturbed genes, keep calculating transitions, for perturbed ones, keep referring to these original states

    #initPerturbed should be a boolean vector of length n
    # If True, gene g will not be updated (perturbation), if False, gene g will be updated starting from biased Ry state normally

    if Tmax==0:
        InitCircuit.measure(list(range(0, n)), list(range(0, n)))
        Aer.backends()
        result = execute(InitCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
        weightDict = result.get_counts(InitCircuit)
        return (outputformat(weightDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None))


    indices2compose = list(range(0, 2*n)) #list [0,...,2n-1], correct
    #print("indices2compose = " + str(indices2compose))
    MultiStepCircuit = InitCircuit.compose(circuit, indices2compose, list(range(0, n))) #InitCircuit+first transition step done, t+1 outputs are at qn to q2n-1
    #BigCirc.compose(smallcirc, qubitindicesOfBigCirc where smallcircqubits should go (length=sizeSmallCirc) )

    #Synchronous circuits, construct once, measure nrshots times
    for t in range(2, Tmax+1): #transitions nr 2-Tmax
        #print("TRANSITION t=" + str(t) + " OUT OF T=" + str(Tmax))
        #following transitions, was range 1 to T+1 before, may need to adapt edge cases
        #circuit has 2n register, attach first n of these to take previous outputs (t-1)*n,...,(t*n)-1 as inputs
        #                         attach second n of these to following qubits (until now carrying default |0>) q_t*n to q_(t+1)*n-1 as new outputs

        indices2compose = list(range((t-1)*n,(t+1)*n)) #(t-1)*n to (t+1)*n-1, 2n qubits in total to compose
        #indices2compose = [None] * n
        for g in range(n): #only need to update first n out of list of length 2n
            #if ( (thetatype == "linear" and initActivities[g] == 0.5) or (thetatype == "angle" and initActivities[g] == 90) ):
            if initPerturbed[g] == False:
                #unperturbed case
                pass
            else:
                #perturbed case
                #print("Overwriting index g=" + str(g) + " (0-indexed)")
                indices2compose[g] = g

        MultiStepCircuit = MultiStepCircuit.compose(circuit, indices2compose, cr)

    #Changed list of qubits to measure, so no reset+overwrite is needed
    qubits2measure = list(range(Tmax*n, (Tmax+1)*n)) #length n
    for g in list(range(n)):
        #if ((thetatype == "linear" and initActivities[g] != 0.5) or (thetatype == "angle" and initActivities[g] != 90)):
        if initPerturbed[g] == True:
            qubits2measure[g] = g #measure initial superposition instead of final output that was not yet overwritten after last transition

    MultiStepCircuit.measure(qubits2measure, range(n))

    if backend != None and transpileCircuit == False:
        warnings.warn("Provided a backend but circuit is not being transpiled to match this backend!")

    #Transpiling circuit if wanted, i.e. mock backends like FakeLima(), FakeRochester()
    if backend != None:
        if transpileCircuit == True:
            #print("Transpiling circuit to backend " + str(backend))
            #print("Circuit depth before transpilation was " + str(MultiStepCircuit.depth()))
            MultiStepCircuit = transpile(MultiStepCircuit, backend=backend, optimization_level=optimization_level,
                                         approximation_degree=approximation_degree, seed_transpiler=seed_transpiler)
            print("Circuit depth after transpilation is " + str(MultiStepCircuit.depth()))
            print("The gates in the deepest path of the transpiled circuit are:")
            from qiskit.converters import circuit_to_dag
            dag = circuit_to_dag(MultiStepCircuit)
            deepestPathOps = dag.count_ops_longest_path()
            print(deepestPathOps)

            #List all gates in the transpiled circuit if desired
            #for gate in MultiStepCircuit.data:
                #print('\ngate name:', gate[0].name)
                #print('qubit(s) acted on:', gate[1])
                #print('other parameters (such as angles):', gate[0].params)

    #Adding noise if wanted
    #print("The depth of the entire Multi-Step circuit is " + str(MultiStepCircuit.depth()) + ". Its width is " + str(MultiStepCircuit.width()) + ".")
    Aer.backends()
    #if noiseModelOrigin != None:
    if backend != None:

        if addNoise == True:
            #print("Adding noise of backend " + str(backend))
            #provider = IBMQ.load_account()
            #backend = provider.get_backend(noiseModelOrigin)
            #noise_model = NoiseModel.from_backend(backend)
            sim_FakeBackend = AerSimulator.from_backend(backend) #Store device properties of given "FakeBackend()" in a new simulator
            result = sim_FakeBackend.run(MultiStepCircuit, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result() #Use this new simulator
            #result = execute(MultiStepCircuit, backend=Aer.get_backend('qasm_simulator'),
            #                 shots=nrshots, noise_model=noise_model).result()  # execute(circuit, simulator, shots)
    else:
        result = execute(MultiStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)

    countDict = result.get_counts(MultiStepCircuit)
    countDict = outputformat(countDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None)
    if returnCircuitDepth == True:
        return MultiStepCircuit.depth(), countDict
    else:
        return(countDict)

#ASYNCHRONOUS STATE TRANSITIONS
"""
Exact multistep circuit transitions which can distinguish if Ry gates should serve only as initial expression bias or be used for true perturbations.
Also allows for addition of noise models. Performs asynchronous state transitions. 
A total of nrshots measurements are performed, with a fraction fractionOfShotsPerCircuit of these for newly generated circuits with new random update orders for each step.
"""
def allparam_exact_multiTransition_asynchronous(rulestxt, Tmax=2, nrshots=100, fractionOfShotsPerCircuit = 0.1, initActivities=None, initPerturbed=None,
                                   thetatype="linear", normalizeOutput=True, sortOutput=True, noiseModelOrigin=None,
                                                seed_transpiler=None, seed_simulator=None):
    circuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="asynchronous")
    #First generate the MultiStepCircuit:
    n = int(circuit.num_qubits/2)
    totalqubits = (Tmax + 1) * n #Needed 2*n qubits for single transition, now (T+1)*n qubits needed for full circuit
    cr = ClassicalRegister(n)

    if initPerturbed == None:
        initPerturbed = list(np.repeat(False, n))

    if initActivities == None:
        InitCircuit = QuantumCircuit(totalqubits, n)
        #default init all-H
        for q in range(0,n):
            InitCircuit.h(q)
    else:
        InitCircuit = initActivityCircuitGenerator(activities=initActivities, Tmax=Tmax, thetatype=thetatype) #Returns only 2n circuit, not (T+1)*n, need to add (T-1)*n qubits

        #all non 0.5 init genes now had an Ry gate added, no need to expand initActivityCircuitGenerator
        #for non-perturbed genes, keep calculating transitions, for perturbed ones, keep referring to these original states

    #initPerturbed should be a boolean vector of length n
    # If True, gene g will not be updated (perturbation), if False, gene g will be updated starting from biased Ry state normally


    bitstring_key_list = list(np.repeat(None, 2 ** n))
    for i in range(2 ** n):
        bitstring_key_list[i] = str('{0:0' + str(n) + 'b}').format(i)
    unionDict = dict.fromkeys(bitstring_key_list, 0)

    nrCircuits = math.ceil(1/fractionOfShotsPerCircuit)
    nrMeasurements = math.ceil(nrshots*fractionOfShotsPerCircuit)
    for c in range(nrCircuits):
        MultiStepCircuit = copy.deepcopy(InitCircuit) #Need a new starting circuit each time
        print("Circuit nr " + str(c) + "/" + str(nrCircuits))
        for t in range(1, Tmax + 1):  # transitions nr 2-Tmax #Changed to start at 1 so that first step also varies asynchronously every time
            circuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="asynchronous")
            # print("TRANSITION t=" + str(t) + " OUT OF T=" + str(Tmax))
            # following transitions, was range 1 to T+1 before, may need to adapt edge cases
            # circuit has 2n register, attach first n of these to take previous outputs (t-1)*n,...,(t*n)-1 as inputs
            #                         attach second n of these to following qubits (until now carrying default |0>) q_t*n to q_(t+1)*n-1 as new outputs

            indices2compose = list(
                range((t - 1) * n, (t + 1) * n))  # (t-1)*n to (t+1)*n-1, 2n qubits in total to compose
            # indices2compose = [None] * n
            for g in range(n):  # only need to update first n out of list of length 2n
                # if ( (thetatype == "linear" and initActivities[g] == 0.5) or (thetatype == "angle" and initActivities[g] == 90) ):
                if initPerturbed[g] == False:
                    # unperturbed case
                    pass
                else:
                    # perturbed case
                    # print("Overwriting index g=" + str(g) + " (0-indexed)")
                    indices2compose[g] = g

            MultiStepCircuit = MultiStepCircuit.compose(circuit, indices2compose, cr)

        # Changed list of qubits to measure, so no reset+overwrite is needed
        qubits2measure = list(range(Tmax * n, (Tmax + 1) * n))  # length n
        for g in list(range(n)):
            # if ((thetatype == "linear" and initActivities[g] != 0.5) or (thetatype == "angle" and initActivities[g] != 90)):
            if initPerturbed[g] == True:
                qubits2measure[g] = g  # measure initial superposition instead of final output that was not yet overwritten after last transition

        MultiStepCircuit.measure(qubits2measure, range(n))

        Aer.backends()
        if noiseModelOrigin != None:
            print("Adding noise!")
            provider = IBMQ.load_account()
            backend = provider.get_backend(noiseModelOrigin)
            noise_model = NoiseModel.from_backend(backend)
            result = execute(MultiStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                             shots=nrMeasurements, noise_model=noise_model, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
        else:
            result = execute(MultiStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                             shots=nrMeasurements, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)


        countDict = result.get_counts(MultiStepCircuit)
        #Add countDict keys to unionDict for every new random update+single measurement
        for k in countDict:
            unionDict[k] = unionDict[k] + countDict[k]


    #Remove zero-count keys from unionDict, normalize if needed, return unionDict
    unionDict = {k: v for k, v in unionDict.items() if v != 0}
    unionDict = outputformat(unionDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None)
    return(unionDict)

#SYNCHRONOUS STATE TRANSITIONS USING REINITIALISATION
"""
Multiple single step transitions, breaks superposition by measuring after every step. Initializes a new statevector according to frequency/weights of measured outcomes.
Version of reinitialising repeated single step circuits which can distinguish if Ry gates should serve only as initial expression bias or be used for true perturbations.
Also allows for addition of noise models. Performs synchronous state transitions.
"""
def allparam_multiTransition_synchronous(rulestxt, Tmax=2, nrshots=100, initActivities=None, initPerturbed=None,
                    thetatype="angle", noiseModelOrigin=None,
                    normalizeOutput=True, sortOutput=True, seed_transpiler=None, seed_simulator=None):
    circuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous")
    n = circuit.num_qubits // 2
    totalcircuitdepth = 0

    if initActivities == None:
        InitCircuit = QuantumCircuit(2*n, n)
        # default init all-H
        for q in range(0, n):
            InitCircuit.h(q)
    else:
        #InitCircuit = initActivityCircuitGenerator(activities=initActivities)
        InitCircuit = initActivityCircuitGenerator(activities=initActivities, Tmax=Tmax, thetatype=thetatype)  # Returns only 2n circuit

    if initPerturbed == None:
        initPerturbed = list(np.repeat(False, n))

    if Tmax == 0:
        InitCircuit.measure(list(range(0, n)), list(range(0, n)))
        Aer.backends()
        result = execute(InitCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots).result()  # execute(circuit, simulator, shots)
        countDict = result.get_counts(InitCircuit)
        return (countDict)

    FirstStepCircuit = InitCircuit.compose(circuit, list(range(0, 2 * n)), list(range(0, n)))
    FirstStepCircuit.measure(list(range(n, 2 * n)), list(range(0, n)))

    # Perform first state transition with Init+circuit
    Aer.backends()
    if noiseModelOrigin != None:
        provider = IBMQ.load_account()
        backend = provider.get_backend(noiseModelOrigin)
        noise_model = NoiseModel.from_backend(backend)
        result = execute(FirstStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots, noise_model=noise_model, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
    else:
        result = execute(FirstStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
    countDict = result.get_counts(FirstStepCircuit)

    totalcircuitdepth += FirstStepCircuit.depth()

    # Loop from T=2,...,T with circuit given previously obtained weights
    for T in range(2, Tmax + 1):
        q_init = QuantumRegister(2 * n)
        c_init = ClassicalRegister(n)
        qc_init = QuantumCircuit(q_init, c_init)
        desired_vector = list(np.array(np.repeat(0, 2 ** n), dtype=complex))
        for key in countDict.keys():
            key2int = int(key, 2)
            desired_vector[key2int] = countDict[key]  # write shot value directly, normalise afterwards

        desired_vector = [np.sqrt(i) for i in desired_vector]  # Need to take square roots since countDict results are measurement results, ie *square* amplitudes
        desired_vector = desired_vector / np.linalg.norm(desired_vector)  # all entries for remaining states have been added, normalise vector
        for i in range(0, len(desired_vector)):
            desired_vector[i] = desired_vector[i] * complex(1, 0) #entries should be complex valued -> multiply by (1+0*i)

        # print("Desired_vector at T = ",T)
        # print(desired_vector)

        # Generalised init of qubits 0-(n-1) to the specified desired_vector, independent of n
        registerstring = ""
        for qindex in range(n):
            registerstring += "q_init[" + str(qindex) + "], "
        registerstring = registerstring[:-2]
        exec("qc_init.initialize(desired_vector, [" + registerstring + "])")

        CompCirc = qc_init.compose(circuit, list(range(0, 2 * n)), list(range(0, n)))

        #Only performing one time step. Check which genes are perturbed and measure from first n qubits, if not perturbed measure from n-2n range
        qubits2measure = list(np.repeat(None, n))
        for g in range(n):
            if initPerturbed[g]:
                qubits2measure[g] = g
            else:
                qubits2measure[g] = g+n

        #measured list(range(n, 2 * n) before
        CompCirc.measure(qubits2measure, list(range(0, n)))
        totalcircuitdepth += CompCirc.depth()

        if noiseModelOrigin != None:
            provider = IBMQ.load_account()
            backend = provider.get_backend(noiseModelOrigin)
            noise_model = NoiseModel.from_backend(backend)
            result = execute(CompCirc, backend=Aer.get_backend('qasm_simulator'),
                             shots=nrshots, noise_model=noise_model, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
        else:
            result = execute(CompCirc, backend=Aer.get_backend('qasm_simulator'),
                             shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
        countDict = result.get_counts(CompCirc)

    countDict = outputformat(countDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None)
    #print("The cumulative depth of the re-initialised single-step circuits is " + str(totalcircuitdepth) + ".")
    return (countDict)

#GROVER AMPLITUDE AMPLIFICATION
"""
Helper function for Grover state transition inversion. 
Returns transition circuit for multiple transition steps without Hadamard initialization or measurement so that circuit can be used as a gate.
"""
def generate_exact_multiTransitionGate(rulestxt, Tmax=1):
    #Get multistep circuit without init, measurement or classical register from new functions so that it can be turned into a gate
    circ = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous", includeClassicalRegister=False)
    n = circ.num_qubits//2 #number nodes in network
    finalcirc = QuantumCircuit(n*(Tmax+1))
    for step in range(1,(Tmax+1)):
        finalcirc = finalcirc.compose(circ, list(range(n*(step-1), n*(step+1))))
    #print(finalcirc)
    return finalcirc

"""
Function which implements G iterations of a Grover Oracle and Diffuser operator.
The marked state (i.e. an attractor) is the solution element. Grover search will then amplify probabilities of states which lie nrTransitions step before the marked state in the STG.
That is, it effectly performs nrTransitions 'inverse' state transitions.
The optimal value for G will depend on the (unknown) number of predecessor states.
"""
def generate_groverSTGinversion_circuit(rulestxt, nrTransitions=1, markedState=None, G=1):
    #markedState should be a list of 0s and 1s of length n

    #Transition circuits as gates to be used as blocks in Grover circuit
    transitionCircuit = generate_exact_multiTransitionGate(rulestxt, Tmax=nrTransitions)
    transitiongate = transitionCircuit.to_gate()
    transitiongate_inv = transitiongate.inverse()

    n = transitionCircuit.num_qubits//(nrTransitions+1)
    GroverCircuit = QuantumCircuit(transitionCircuit.num_qubits + 1, n)
    minusqubitindex = transitionCircuit.num_qubits

    # Put last qubit in |minus> state via X->H
    GroverCircuit.x(minusqubitindex)
    GroverCircuit.h(minusqubitindex)

    # Init genes with H layer
    for q in range(n):
        GroverCircuit.h(q)

    markedState = markedState[::-1]
    for GroverIteration in range(G):
        ### ORACLE
        # Transition gate block
        GroverCircuit.append(transitiongate, list(range(n*(nrTransitions+1))))

        # Loop for X gates corresponding to markedState on output register
        outputregister = list(range(n*nrTransitions, n*(nrTransitions+1)))

        for g in range(n):
            #need X gate if there is a 0 in markedState, no X gate if there is a 1
            if markedState[g] == 0:
                GroverCircuit.x(outputregister[g])

        # MCX gate over output of transition circuit, target |minus> ancilla
        GroverCircuit.mcx(outputregister, minusqubitindex)

        # Loop for X gates corresponding to markedState
        for g in range(n):
            #need X gate if there is a 0 in markedState, no X gate if there is a 1
            if markedState[g] == 0:
                GroverCircuit.x(outputregister[g])

        # Inverse Transition gate block
        GroverCircuit.append(transitiongate_inv, list(range(n * (nrTransitions + 1))))

        ###DIFFUSER
        # Hadamard layer for initital genes
        for q in range(n):
            GroverCircuit.h(q)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # MCX gates over initial genes, target |minus> ancilla
        GroverCircuit.mcx(list(range(n)), minusqubitindex)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # Hadamard layer for initial genes
        for q in range(n):
            GroverCircuit.h(q)


    # Add measurement operators for initial gene qubits
    GroverCircuit.measure(list(range(n)), list(range(n)))

    return(GroverCircuit)



#QUANTUM COUNTING
"""
Function to translate measured bitstrings from Quantum Counting circuit into probabilities for M as the number of predecessor states with a t qubit readout register.
"""
def calculateMfromBitstring(outputDict, t, n, verbose=True):
    measured_int_dict = {int(k,2):float(v) for k,v in outputDict.items()} #dict with keys = integers, values = floats/probabilities
    # int(k,2) counts e.g. 01101 as 13 -> q0 i.e. least valued bit is written at the right
    N = 2**n
    theta_dict = {(k*2*np.pi/(2**t)):float(v) for k,v in measured_int_dict.items()} #dict with keys = phase angles
    NminusM_dict = {(N*np.sin(k/2)*np.sin(k/2)):float(v) for k,v in theta_dict.items()} #dict with keys = N-M values
    M_dict = {(N-k):float(v) for k,v in NminusM_dict.items()}
    #M_dict Returns non-solutions instead of solutions since implemented diffuser -U_s instead of U_s -> Need to calculate M = N - returnedValue
    solutions_dict = outputformat({round(k):float(v) for k,v in M_dict.items()}) #returns solutions M, rounded to integers to aggregate some keys and their weights together

    weightedAverage = 0
    for k in solutions_dict.keys():
        weightedAverage += k*solutions_dict[k]

    solutions_array = [0] * (N + 1) #0-N are all possible solutions
    for k in solutions_dict.keys():
        solutions_array[k] = solutions_dict[k]

    if verbose:
        print("The weighted average of the returned M dictionary is " + str(round(weightedAverage,3)))
        print("The standard deviation of the measured distribution is " + str(round(np.std(solutions_array),3)))
        print("The most likely measurement of the circuit corresponds to M = " + str(max(solutions_dict, key=solutions_dict.get)))
    return(solutions_dict)


"""
Function which generates the Grover operator G for a given number of inverted transitions and a marked state of interest.
Returns a circuit (without initialization gates) which can then be exponentiated and controlled for use in the Quantum Counting algorithm.
"""
def GroverIterationFromTransitionCircuit(transitionCircuit, markedState, nrTransitions):
    transitiongate = transitionCircuit.to_gate()
    transitiongate_inv = transitiongate.inverse()

    n = transitionCircuit.num_qubits // (nrTransitions + 1)
    GroverCircuit = QuantumCircuit(transitionCircuit.num_qubits + 1)
    minusqubitindex = transitionCircuit.num_qubits

    ### ORACLE
    # Transition gate block
    GroverCircuit.append(transitiongate, list(range(n * (nrTransitions + 1))))

    # Loop for X gates corresponding to markedState on output register
    outputregister = list(range(n * nrTransitions, n * (nrTransitions + 1)))
    markedState = markedState[::-1]

    for g in range(n):
        # need X gate if there is a 0 in markedState, no X gate if there is a 1
        if markedState[g] == 0:
            GroverCircuit.x(outputregister[g])

    # MCX gate over output of transition circuit, target |minus> ancilla
    GroverCircuit.mcx(outputregister, minusqubitindex)

    # Loop for X gates corresponding to markedState
    for g in range(n):
        # need X gate if there is a 0 in markedState, no X gate if there is a 1
        if markedState[g] == 0:
            GroverCircuit.x(outputregister[g])

    # Inverse Transition gate block
    GroverCircuit.append(transitiongate_inv, list(range(n * (nrTransitions + 1))))

    ###DIFFUSER
    # Hadamard layer for initital genes
    for q in range(n):
        GroverCircuit.h(q)

    # X layer for initial genes
    for q in range(n):
        GroverCircuit.x(q)

    # MCX gates over initial genes, target |minus> ancilla
    GroverCircuit.mcx(list(range(n)), minusqubitindex)

    # X layer for initial genes
    for q in range(n):
        GroverCircuit.x(q)

    # Hadamard layer for initial genes
    for q in range(n):
        GroverCircuit.h(q)

    return(GroverCircuit)


"""
Generate circuit for a quantum counting algorithm with a readout register of r_registerLen qubits. Performs nrTransitions inverted transitions from the markedState.
Returns a dictionary with probabilities for the resulting number of predecessor states M (rounded to integers).
Increasing the size of the readout register will yield more accurate results.
"""
def QuantumCountingAlgo(rulestxt, nrTransitions, markedState, r_registerLen, nrshots = 1000, do_swaps_QFT = True, approxmation_degree_QFT = 0,
                        verbose=True, seed_transpiler=None, seed_simulator=None):
    # Full QCounting circ generation U^2^(t-1), U^2^(t-2)...U^2^0 -> QFTdgr -> measure r register -> resulting state bitstring encodes |2^r ANGLE>
    transitionCircuit = generate_exact_multiTransitionGate(rulestxt, Tmax=nrTransitions)
    G = GroverIterationFromTransitionCircuit(transitionCircuit, markedState, nrTransitions)
    n = transitionCircuit.num_qubits // (nrTransitions+1)
    QCountingCirc = QuantumCircuit((nrTransitions+1)*n+1+r_registerLen, r_registerLen) #full circuit with measurement + Grover registers
    MeasurementRegister = list(range(r_registerLen))
    GroverRegister = list(range(r_registerLen, QCountingCirc.num_qubits))
    if verbose:
        print("Constructing Quantum Counting Circuit with a total of " + str(QCountingCirc.num_qubits) + " qubits.")
        #print("G circuit to be used as a module:")
        #print(G)

    #Initialize MeasurementRegister with H gates as well
    for q in MeasurementRegister:
        QCountingCirc.h(q)
    #for q in GroverRegister:
    for q in GroverRegister[:n]: #Transition output register(s) of Grover should not be initialised with H -> input n genes need H layer but output should be mapped onto 0 qubits!!
        QCountingCirc.h(q)
    #for q in GroverRegister[:-1]: #No H gate for last qubit which is the minus state
    #    QCountingCirc.h(q)

    #Final qubit initialised as |minus> once outside of all cGrover iterators
    QCountingCirc.x(QCountingCirc.num_qubits-1)
    QCountingCirc.h(QCountingCirc.num_qubits - 1)

    #Compose onto circuit in a loop with t register as control qubits
    for t in range(r_registerLen):
        powerof2Gcircuit = QuantumCircuit(G.num_qubits)
        powerof2Gcircuit = powerof2Gcircuit.compose(G, list(range(G.num_qubits)))
             #for t=0 -> return this immediately
        for compose in range(2**t - 1):
            #for t=1 -> G^2^1 = G^2 compose 1 more time
            #for t=2 -> G^2^2 = G^4 compose 3 more times --> compose 2^t-1 more times G after G into
            powerof2Gcircuit = powerof2Gcircuit.compose(G, list(range(G.num_qubits)))

        powerof2Ggate = powerof2Gcircuit.to_gate()
        powerof2cGgate = powerof2Ggate.control(num_ctrl_qubits=1)
        #NOTE: Control order matters! Need q0 controlling G^2^0, q1 controlling G^2^1 etc. -> otherwise results are nonsense
        cGroverRegister = [t] + GroverRegister #append to beginning of list, G^2^0 controlled by q0
        #cGroverRegister = [t_registerLen-t-1] + GroverRegister  # append to beginning of list, G^2^0 controlled by q (t_registerLen-1)
        QCountingCirc = QCountingCirc.compose(powerof2cGgate, cGroverRegister)

    from qiskit.circuit.library import QFT
    QFTdgrCirc = QFT(num_qubits=r_registerLen, approximation_degree=approxmation_degree_QFT, do_swaps=do_swaps_QFT, inverse=True)
    # Test if do_swaps can be set to False without changing results, i.e. test if classical instead of quantum reordering is already done inside the QFT function

    QCountingCirc = QCountingCirc.compose(QFTdgrCirc, list(range(r_registerLen)))
    QCountingCirc.measure(MeasurementRegister, MeasurementRegister)
    #print("Final QCountingCirc:")
    #print(QCountingCirc)

    if verbose:
        #print("CountOps CX gates of final circuit:")
        from qiskit.test.mock import FakeToronto
        #transpiledCircuit = transpile(QCountingCirc, backend=FakeToronto(), optimization_level=3, seed_transpiler=1234)
        #print(transpiledCircuit.count_ops())
        print("QCountingCirc depth = " + str(QCountingCirc.depth) + ", width = " + str(QCountingCirc.num_qubits) + ".")
        #--> 34950 CX gates for 3 controlled grover gates for 1 transition in Giaco + IQFT

    result = execute(QCountingCirc, backend=Aer.get_backend('qasm_simulator'), shots=nrshots,
                     seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()
    measured_int_dict = result.get_counts(QCountingCirc)
    M_dict = calculateMfromBitstring(measured_int_dict, r_registerLen, n, verbose)
    return(M_dict)




#MULTI STATE SUPPRESSION FUNCTION:
#markedState always a list of lists, can take multiple marked states
def generate_groverSTGdeletion_circuit_multimarkedState(rulestxt, nrTransitions=1, markedState=None, G=1, M=1):
    # markedState should be a list of 0s and 1s of length N

    # Transition circuits as gates to be used as blocks in Grover circuit
    transitionCircuit = generate_exact_multiTransitionGate(rulestxt, Tmax=nrTransitions)
    transitiongate = transitionCircuit.to_gate()
    transitiongate_inv = transitiongate.inverse()

    n = transitionCircuit.num_qubits // (nrTransitions + 1)
    N = 2**n

    #Get adapted angle phi (lam here) for deletion
    betacos = np.sqrt((N - M) / N) #angle quantifies overlap with number of solutions/marked states M
    lam = 2 * np.arcsin(1 / (2 * betacos)) #lambda, adapted phi angle for deletion
    #Converges to pi/3 for deletion of exactly 1 state out of N (Liu paper)

    #TODO: Debug warning for arcsin range
    #arcsin argument should not be larger than 1 -> need 2*betacos > 1 -> TODO: Should also not be below -1
    #-> need sqrt((N-M)/N) > 0.5
    #-> (N-M)/N > 0.25 -> (N-M) = nr unmarked > 0.25*N

    GroverCircuit = QuantumCircuit(transitionCircuit.num_qubits + 1, n)
    minusqubitindex = transitionCircuit.num_qubits

    # Put last qubit in |minus> state via X->H
    GroverCircuit.x(minusqubitindex)
    GroverCircuit.h(minusqubitindex)

    # Init genes with H layer
    for q in range(n):
        GroverCircuit.h(q)

    #markedState = markedState[::-1]
    markedState = [state[::-1] for state in markedState] #invert all the states in the List of Lists
    for GroverIteration in range(G):
        ### ORACLE
        # Transition gate block
        GroverCircuit.append(transitiongate, list(range(n * (nrTransitions + 1))))

        # Loop for X gates corresponding to markedState on output register
        outputregister = list(range(n * nrTransitions, n * (nrTransitions + 1)))

        #TODO: Added for loop over all elements in markedState, add multiple phase kickbacks in oracle
        for ms in markedState:
            GroverCircuit.barrier()
            for g in range(n):
                # need X gate if there is a 0 in markedState, no X gate if there is a 1
                if ms[g] == 0:
                    GroverCircuit.x(outputregister[g])

            # MCP gate over output of transition circuit, target |minus> ancilla
            # Changed for deletion to not do a 180° phase flip with x instead, but a phase flip dependent on M
            GroverCircuit.mcp(-lam, outputregister, minusqubitindex)

            # Loop for X gates corresponding to markedState
            for g in range(n):
                # need X gate if there is a 0 in markedState, no X gate if there is a 1
                if ms[g] == 0:
                    GroverCircuit.x(outputregister[g])
            GroverCircuit.barrier()

        # Inverse Transition gate block
        GroverCircuit.append(transitiongate_inv, list(range(n * (nrTransitions + 1))))

        # GLOBAL PHASE SHIFT
        #GroverCircuit.p(lam, list(range(n)))
        #GroverCircuit.x(list(range(n)))
        #GroverCircuit.p(lam, list(range(n)))
        #GroverCircuit.x(list(range(n)))


        ###DIFFUSER
        # Hadamard layer for initital genes
        for q in range(n):
            GroverCircuit.h(q)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # MCX gates over initial genes, target |minus> ancilla
        GroverCircuit.mcp(+lam, list(range(n)), minusqubitindex)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # Hadamard layer for initial genes
        for q in range(n):
            GroverCircuit.h(q)

    # Add measurement operators for initial gene qubits
    GroverCircuit.measure(list(range(n)), list(range(n)))

    return (GroverCircuit)



