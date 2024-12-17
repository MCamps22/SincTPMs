'''
Copyright 2024 Michel Campos

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import random, math
import numpy as np
from datetime import datetime
import csv
import hashlib
import json

"Parameters of a TPM: "
"k : number of hidden units"
"n: number of inputs neurons"
"l: range of discrete weigths"


def innerProduct(v, u):
    aux = 0
    for i in range(len(v)):
        aux = aux + v[i] * u[i]
    return aux

def areDifferentFunction(v, u):
    aux = 0
    for i in range(len(v)):
        if v[i] == u[i]:
            aux = aux + 1
    if aux == len(v):
        return False
    else:
        return True

def stimulusGeneratorNonBinary():
   return random.choice([-3, -2, -1, 1, 2, 3])

def stimulusGeneratorBinary():
    if random.randint(0, 1) == 0:
        return -1
    else:
        return 1


def stimulusVectorGenerator(n,stimulusBinary):
    estimulus = []
    if stimulusBinary:
        for i in range(n):
            estimulus = estimulus + [stimulusGeneratorBinary()]
    else:
        for i in range(n):
            estimulus = estimulus + [stimulusGeneratorNonBinary()]
    return estimulus

def weigthGerator(l):
    return random.randint(-l, l)

def weigthsVectorhGerator(n, l):
    weigth = []
    for i in range(n):
        weigth = weigth + [weigthGerator(l)]
    return weigth

def sign(r):
    if r > 0:
        return 1
    elif r == 0 or r < 0:
        return -1

def heavisideStepFunction(x):
    if x > 0:
        return 1
    else:
        return 0

def gFunction(x, l):
    if abs(x) > l:
        return sign(x) * l
    else:
        return x

def localFieldFunction(n, v, u):
    return (innerProduct(v, u) / (math.sqrt(n)))

def outputSigmaFunction(h):
    if h == 0:
        return -1
    else:
        return sign(h)

def hebbianLearningRule(tA, tB, stA, wsA):
    # tA, tB are the outputs of TPM A and TPM B, respectively.
    # stA is the stimulus vector, and wsA is the weight vector of TPM A.
    sigma = outputSigmaFunction(localFieldFunction(n, stA, wsA))
    for i in range(len(wsA)):
        wsA[i] = gFunction(wsA[i] + stA[i] * tA * heavisideStepFunction(sigma * tA) * heavisideStepFunction(tA * tB), l)
    return wsA

def antiHebbianLearningRule(tA, tB, stA, wsA):
    sigma = outputSigmaFunction(localFieldFunction(n, stA, wsA))
    for i in range(len(wsA)):
        wsA[i] = gFunction(wsA[i] - stA[i] * tA * heavisideStepFunction(sigma * tA) * heavisideStepFunction(tA * tB), l)
    return wsA

def randomWalkLearningRule(tA, tB, stA, wsA):
    sigma = outputSigmaFunction(localFieldFunction(n, stA, wsA))
    for i in range(len(wsA)):
        wsA[i] = gFunction(wsA[i] + stA[i] * heavisideStepFunction(sigma * tA) * heavisideStepFunction(tA * tB), l)
    return wsA
# The thauFunction(x) function returns the output of the entire TPM network.
def thauFunction(x):
    mul = 1
    for i in range(len(x)):
        mul = mul * x[i]
    return mul

def switch_case_rule(argument):
    
    if argument == 1:
        return "Hebbian Learning"
    elif argument == 2:
        return "AntiHebbian Learning"
    elif argument == 3:
        return "Random-Walk Learning"
    else:
        return "Invalid option"

def add_row_to_csv(file_path, row_data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)
    

def generate_matrix_and_vector(weitTPM, valueL):
    # Generate matrix A
    matrix = []
    for i in range(len(weitTPM[0])):
        row = []
        for j in range(len(weitTPM) - 1):
            row.append(weitTPM[j][i])
        matrix.append(row)

    # Generate VECTOR A
    vector = []
    for value in weitTPM[-1]:
        if value == 0:
            value = valueL + 1
        vector.append([value])
        
    vector_transpose = np.transpose(vector)
  
    return matrix, vector_transpose


#Polinomial
def get_k_values(N, K):
    len_K = max(N, K)
    k_values = list(range(1, len_K + 1)) 
    return k_values

def calculate_P(K, N, weights, k_values):
    P = 0
    for i in range(K):
        inner_sum = 0
        for j in range(N):

            inner_sum += weights[i][j] ** (2 * k_values[j] + 1)
    
        P += inner_sum ** (2 * k_values[i] + 1)
    return P

#--- Main ---



#Input of the architecture to be used in the simulation
print('Enter the parameters of TPMs....')
print('k : number of hidden units')
k = int(input())
print('n: number of inputs neurons')
n = int(input())
print('l: range of discrete weigths')
l = int(input())

use_stimulus_binary = True
stimulus_type = input("Do you want to use binary stimuli (only -1 or 1)? (Yes[Y] / No[N]): ").strip().lower()
if stimulus_type == 'y':
    use_stimulus_binary = True
    print("Binary stimuli selected: -1 or 1.")
else:
    use_stimulus_binary = False
    print("Non-binary stimuli selected: -3, -2, -1, 1, 2, 3")




#Global Variables
now = datetime.now()
timestamp = now.strftime("%d%m%Y_%H%M%S")
countSync=0
list_iteraciones_por_Sync=[]
list_ajustes_por_Sync=[]
list_falsosPositivos_por_sync=[]
list_falsosNegativos_por_Sync=[]
weitTPMA = []
weitTPMB = []
totalAdjustmentsCount=0
totalFalsePositivesCount=0
totalIterationsCount=0
totalFalseNegativesCount=0

#Matrix Variables

matriz_a = np.array([[0]])
matriz_b = np.array([[0]])



#Input learning rule
print('Select learning rule:')
print(' 1. Hebbian Learning Rule','2. AntiHebbian Learning Rule', '3. Random-Walk Learning Rule', sep = '   ' )

chosenLearRule = int(input())
nameRule= switch_case_rule(chosenLearRule)


print('Select verification algorithm:')
print('1. Polinomial','2. Hash',' 3. Matriz ', sep = '   ' )
chosenAlgorithm = int(input())

print('Empty weitTPMA = ', weitTPMA)
print('Empty weitTPMB = ', weitTPMB)





# The following function is the implementation of the synchronization for two TPMs,
# which has k hidden neurons with n inputs.
def synchroTPMs(k, n, l, learRule,weitTPMA, weitTPMB , countSync):
    sigmaA = []
    sigmaB = []
    stimuTPMA = []
    stimuTPMB = []

    numAjustes=0
    numFalsosPositivos=0
    numFalsosNegativos=0
    
    # The following cycle generates the random stimulus vectors
    for i in range(k):
        stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
    stimuTPMB = stimuTPMA
    # The following cycle generates the random weigths vectors
    for i in range(k):
        weitTPMA = weitTPMA + [weigthsVectorhGerator(n, l)]
        weitTPMB = weitTPMB + [weigthsVectorhGerator(n, l)]
    
    print("Using matrix-based verification method")
    print('Initial stimuTPMA = ', stimuTPMA)
    print('Initial stimuTPMB = ', stimuTPMB)
    print('Initial weitTPMA = ', weitTPMA)
    print('Initial weitTPMB = ', weitTPMB)
    
   
    for i in range(k):
        sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
        sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))

    thauA = thauFunction(sigmaA)
    thauB = thauFunction(sigmaB)
  
    j = 0
    while areDifferentFunction(weitTPMA, weitTPMB):
        matriz_a, vector_a = generate_matrix_and_vector(weitTPMA,l)
        matriz_b, vector_b = generate_matrix_and_vector(weitTPMB,l)
        resMatrizTpmA = np.dot(vector_a,matriz_a )
        resMatrizTpmB = np.dot(vector_b,matriz_b )
        if np.array_equal(resMatrizTpmA,resMatrizTpmB):
            numFalsosPositivos= numFalsosPositivos+1
        
        if thauA == thauB:  # The weights must be adjusted according to the used learning rule.
            numAjustes=numAjustes+1
            for i in range(k):
                if sigmaA[i] == thauA:
           
                    if learRule == 1:
                        weitTPMA[i] = hebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    elif learRule == 2:
                        weitTPMA[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    else:
                        weitTPMA[i] = randomWalkLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                if sigmaB[i] == thauB:
                  
                    if learRule == 1:
                        weitTPMB[i] = hebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    elif learRule == 2:
                        weitTPMB[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    else:
                        weitTPMB[i] = randomWalkLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])

        stimuTPMA = []
        stimuTPMB = []
        # The following cycle generates the random stimulus vectors
        for i in range(k):
            stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
            stimuTPMB = stimuTPMA
        sigmaA = []
        sigmaB = []
        # The following cycle generates the outputs of each hidden neuron
        for i in range(k):
            sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
            sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))
        # The following variables contains the total outputs of each net.
        thauA = thauFunction(sigmaA)
        thauB = thauFunction(sigmaB)
        j = j + 1
        
    
     
    print('---- Iterations number = ', j)
    print('Final weitTPMA = ', weitTPMA)
    print('Final weitTPMB = ', weitTPMB)
    print('General Adjustments = ',numAjustes)
    
    matriz_a, vector_a = generate_matrix_and_vector(weitTPMA,l)
    matriz_b, vector_b = generate_matrix_and_vector(weitTPMB,l)
    resMatrizTpmA = np.dot(vector_a,matriz_a)
    resMatrizTpmB = np.dot(vector_b,matriz_b)
    
    print("Result of TPM A verification method:" , resMatrizTpmA)
    print("Result of TPM B verification method:" , resMatrizTpmB)
    
    if not(np.array_equal(resMatrizTpmA,resMatrizTpmB)):
        numFalsosNegativos+=1
        
    return weitTPMA,numAjustes,numFalsosPositivos,j,numFalsosNegativos

def synchroTPMsHash(k, n, l, learRule,weitTPMA, weitTPMB , countSync):
    sigmaA = []
    sigmaB = []
    stimuTPMA = []
    stimuTPMB = []

    numAjustes=0
    numFalsosPositivos=0
    numFalsosNegativos=0
    
    # The following cycle generates the random stimulus vectors
    for i in range(k):
        stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
    stimuTPMB = stimuTPMA
    # The following cycle generates the random weigths vectors
    for i in range(k):
        weitTPMA = weitTPMA + [weigthsVectorhGerator(n, l)]
        weitTPMB = weitTPMB + [weigthsVectorhGerator(n, l)]
    
    print("Using Hash verification method")
    print('Initial stimuTPMA = ', stimuTPMA)
    print('Initial stimuTPMB = ', stimuTPMB)
    print('Initial weitTPMA = ', weitTPMA)
    print('Initial weitTPMB = ', weitTPMB)
   
    for i in range(k):
        sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
        sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))

    thauA = thauFunction(sigmaA)
    thauB = thauFunction(sigmaB)
  
    j = 0
    while areDifferentFunction(weitTPMA, weitTPMB):


        listA_json = json.dumps(weitTPMA)
        hash_objectA = hashlib.sha1()
        hash_objectA.update(listA_json.encode('utf-8'))
        hash_hexA = hash_objectA.hexdigest() 

        listB_json = json.dumps(weitTPMB)
        hash_objectB = hashlib.sha1()
        hash_objectB.update(listB_json.encode('utf-8'))
        hash_hexB = hash_objectB.hexdigest()
        
        if  hash_hexA ==  hash_hexB:
            numFalsosPositivos= numFalsosPositivos+1

        if thauA == thauB:  # The weights must be adjusted according to the used learning rule.
            numAjustes=numAjustes+1
            for i in range(k):
                if sigmaA[i] == thauA:
           
                    if learRule == 1:
                        weitTPMA[i] = hebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    elif learRule == 2:
                        weitTPMA[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    else:
                        weitTPMA[i] = randomWalkLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                if sigmaB[i] == thauB:
                  
                    if learRule == 1:
                        weitTPMB[i] = hebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    elif learRule == 2:
                        weitTPMB[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    else:
                        weitTPMB[i] = randomWalkLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])

        stimuTPMA = []
        stimuTPMB = []
        # The following cycle generates the random stimulus vectors
        for i in range(k):
            stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
            stimuTPMB = stimuTPMA
        sigmaA = []
        sigmaB = []
        # The following cycle generates the outputs of each hidden neuron
        for i in range(k):
            sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
            sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))
        # The following variables contains the total outputs of each net.
        thauA = thauFunction(sigmaA)
        thauB = thauFunction(sigmaB)
        j = j + 1
        
    
     
    print('---- Iterations number = ', j)
    print('Final weitTPMA = ', weitTPMA)
    print('Final weitTPMB = ', weitTPMB)
    print('General Adjustments = ',numAjustes)
    
    listA_json = json.dumps(weitTPMA)
    hash_objectA = hashlib.sha1()
    hash_objectA.update(listA_json.encode('utf-8'))
    hash_hexA = hash_objectA.hexdigest()

    listB_json = json.dumps(weitTPMB)
    hash_objectB = hashlib.sha1()
    hash_objectB.update(listB_json.encode('utf-8'))
    hash_hexB = hash_objectB.hexdigest()
    
    print("Result of TPM A verification method:" , hash_hexA)
    print("Result of TPM B verification method:" , hash_hexB)
    
    if not(hash_hexA ==  hash_hexB):
        numFalsosNegativos+=1
        
    return weitTPMA,numAjustes,numFalsosPositivos,j,numFalsosNegativos

def synchroTPMsPolinomial(k, n, l, learRule,weitTPMA, weitTPMB , countSync, k_values):
    sigmaA = []
    sigmaB = []
    stimuTPMA = []
    stimuTPMB = []

    numAjustes=0
    numFalsosPositivos=0
    numFalsosNegativos=0
    
    # The following cycle generates the random stimulus vectors
    for i in range(k):
        stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
    stimuTPMB = stimuTPMA
    # The following cycle generates the random weigths vectors
    for i in range(k):
        weitTPMA = weitTPMA + [weigthsVectorhGerator(n, l)]
        weitTPMB = weitTPMB + [weigthsVectorhGerator(n, l)]
    print("Using Polinomial verification method")
    print('Initial stimuTPMA = ', stimuTPMA)
    print('Initial stimuTPMB = ', stimuTPMB)
    print('Initial weitTPMA = ', weitTPMA)
    print('Initial weitTPMB = ', weitTPMB)
   
    for i in range(k):
        sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
        sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))

    thauA = thauFunction(sigmaA)
    thauB = thauFunction(sigmaB)
  
    j = 0
    while areDifferentFunction(weitTPMA, weitTPMB):
        
        
        #POLINOMIAL TPM A -----
        p_calculate_A=calculate_P(k, n, weitTPMA, k_values)
        
        #POLINOMIAL TPM B -----
        p_calculate_B=calculate_P(k, n, weitTPMB, k_values)
        
        
        if  p_calculate_A ==  p_calculate_B:
            numFalsosPositivos= numFalsosPositivos+1
        
        if thauA == thauB:  # The weights must be adjusted according to the used learning rule.
            numAjustes=numAjustes+1
            for i in range(k):
                if sigmaA[i] == thauA:
           
                    if learRule == 1:
                        weitTPMA[i] = hebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    elif learRule == 2:
                        weitTPMA[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                    else:
                        weitTPMA[i] = randomWalkLearningRule(thauA, thauB, stimuTPMA[i], weitTPMA[i])
                if sigmaB[i] == thauB:
                  
                    if learRule == 1:
                        weitTPMB[i] = hebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    elif learRule == 2:
                        weitTPMB[i] = antiHebbianLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])
                    else:
                        weitTPMB[i] = randomWalkLearningRule(thauA, thauB, stimuTPMB[i], weitTPMB[i])

        stimuTPMA = []
        stimuTPMB = []
        # The following cycle generates the random stimulus vectors
        for i in range(k):
            stimuTPMA = stimuTPMA + [stimulusVectorGenerator(n,use_stimulus_binary)]
            stimuTPMB = stimuTPMA
        sigmaA = []
        sigmaB = []
        # The following cycle generates the outputs of each hidden neuron
        for i in range(k):
            sigmaA.append(outputSigmaFunction(localFieldFunction(n, stimuTPMA[i], weitTPMA[i])))
            sigmaB.append(outputSigmaFunction(localFieldFunction(n, stimuTPMB[i], weitTPMB[i])))
        # The following variables contains the total outputs of each net.
        thauA = thauFunction(sigmaA)
        thauB = thauFunction(sigmaB)
        j = j + 1
        
    
     
    print('---- Iterations number = ', j)
    print('Final weitTPMA = ', weitTPMA)
    print('Final weitTPMB = ', weitTPMB)
    print('General Adjustments = ',numAjustes)
    
    #POLINOMIAL TPM A -----
    p_calculate_A=calculate_P(k, n, weitTPMA, k_values)
    
    #POLINOMIAL TPM B -----
    p_calculate_B=calculate_P(k, n, weitTPMB, k_values)
    
    print("Result of TPM A verification method:" , p_calculate_A)
    print("Result of TPM B verification method:" , p_calculate_B)
    
    
    if not(p_calculate_A ==  p_calculate_B):
        numFalsosNegativos+=1
        
    return weitTPMA,numAjustes,numFalsosPositivos,j,numFalsosNegativos

#Cicle Simulation


if chosenAlgorithm == 1:
    k_values = get_k_values(n,k)
    
for x in range (100):
    countSync=countSync+1
    print("N SINC: ", countSync)
    if chosenAlgorithm == 3:
        v,adjustmentsCount,falsePositivesCount,iterationsCount,falseNegativesCount = synchroTPMs(k, n, l,chosenLearRule,weitTPMA, weitTPMB,countSync)
    elif chosenAlgorithm == 2:
        v,adjustmentsCount,falsePositivesCount,iterationsCount,falseNegativesCount = synchroTPMsHash(k, n, l,chosenLearRule,weitTPMA, weitTPMB,countSync)
    elif chosenAlgorithm == 1:
        v,adjustmentsCount,falsePositivesCount,iterationsCount,falseNegativesCount = synchroTPMsPolinomial(k, n, l,chosenLearRule,weitTPMA, weitTPMB,countSync,k_values)

    list_iteraciones_por_Sync.append(iterationsCount)
    list_ajustes_por_Sync.append(adjustmentsCount)
    list_falsosPositivos_por_sync.append(falsePositivesCount)
    list_falsosNegativos_por_Sync.append(falseNegativesCount )
    totalAdjustmentsCount=totalAdjustmentsCount+adjustmentsCount
    totalFalsePositivesCount=totalFalsePositivesCount+falsePositivesCount
    totalIterationsCount=totalIterationsCount+iterationsCount
    totalFalseNegativesCount=totalFalseNegativesCount+falseNegativesCount 
    



if chosenAlgorithm == 3:
    file_path_resumen = f'Sync_Matrix_k{k}-N{n}-l{l}'
    
    if use_stimulus_binary:
        file_path_resumen += '_binary'
    else:
        file_path_resumen += '_non_binary'
    file_path_resumen += f'-{timestamp}.csv'
    
    file_path_resumen_por_sync = f'Sync_Matrix_x_sync_k{k}-N{n}-l{l}'
    if use_stimulus_binary:
        file_path_resumen_por_sync += '_binary'  
    else:
        file_path_resumen_por_sync += '_non_binary' 
    file_path_resumen_por_sync += f'-{timestamp}.csv'

elif chosenAlgorithm == 2:
    file_path_resumen = f'Sync_Hash_k{k}-N{n}-l{l}'
    
    if use_stimulus_binary:
        file_path_resumen += '_binary'
    else:
        file_path_resumen += '_non_binary'
    file_path_resumen += f'-{timestamp}.csv'
    
    file_path_resumen_por_sync = f'Sync_Hash_x_sync_k{k}-N{n}-l{l}'
    if use_stimulus_binary:
        file_path_resumen_por_sync += '_binary'  
    else:
        file_path_resumen_por_sync += '_non_binary' 
    file_path_resumen_por_sync += f'-{timestamp}.csv'

elif chosenAlgorithm == 1:
    file_path_resumen = f'Sync_Poli_k{k}-N{n}-l{l}'
    
    if use_stimulus_binary:
        file_path_resumen += '_binary'
    else:
        file_path_resumen += '_non_binary'
    file_path_resumen += f'-{timestamp}.csv'
    
    file_path_resumen_por_sync = f'Sync_Poli_x_sync_k{k}-N{n}-l{l}'
    if use_stimulus_binary:
        file_path_resumen_por_sync += '_binary'  
    else:
        file_path_resumen_por_sync += '_non_binary' 
    file_path_resumen_por_sync += f'-{timestamp}.csv'

columns_resumen = ['Total Synchronizations', 'Total Adjustments', 'Total False Positives', 'Total Successes', 'Total Iterations', 'Total False Negatives']

resumen_data = [countSync, totalAdjustmentsCount, totalFalsePositivesCount, countSync, totalIterationsCount, totalFalseNegativesCount]
with open(file_path_resumen, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns_resumen)
    writer.writerow(resumen_data)


# Data for 'Sync' CSV file
rows_resumen_por_sync = zip(
    list_iteraciones_por_Sync, list_ajustes_por_Sync, 
    list_falsosPositivos_por_sync, list_falsosNegativos_por_Sync
)

with open(file_path_resumen_por_sync, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Iterations per Sync", "Adjustments per Sync", "False Positives per Sync", "False Negatives per Sync"])
    writer.writerows(rows_resumen_por_sync)

print("--end of the simulation---")
    
    
