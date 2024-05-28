import math
import random
from utils import primes
import numpy as np
import sympy
import multiprocessing
from time import time
from math import gcd

import numpy
import math
from numpy import linalg

def index_calc(a,b,n, async_=False, process_count=4):
    base = get_base(n-1,3.38)
    
    start = time() 
    print('Trying to find inverse matrix:')  
    inv,adj = get_inverse(base,a,n,async_,process_count)
    print('Matrix found in',time()-start)
    vector = inverse_matrix(inv,sympy.Matrix(adj),n)
    
    
    log = find_log(a,b,n,base,vector)
    return log



def get_inverse(base,a,n,async_,process_count):
    ind = 0
    while True:
        ind+=1
        start = time()
        matrix,adj = get_equations(base,a,n)
        print('Eq fetched in',time()-start)
        try:
            matrix_ = sympy.Matrix(matrix)

            det = matrix_.det()
            inv_det = pow(det,-1,n-1)
            if type(inv_det) != sympy.core.numbers.Integer: continue
            start = time()
            inv = modMatInv(matrix_,n-1,inv_det,async_,process_count)
            
            
            break
        except Exception as e:
            print('Wrong matrix',e)
    return inv,adj
    
def modMatInv(A,p, inv_det,async_,process_count):      
    n=A.rows
    adj = A.copy()
    minors = get_minor_matrix(adj,async_,process_count)
    for i in range(0,n):
        for j in range(0,n):
            adj[j,i]=((-1)**(i+j)*minors[i,j])%p
    return (inv_det*adj)%p

def get_minor_matrix(A, async_=True, workers_count = 8):
    B=A.copy()
    n=A.rows
    
    if async_:
        processes = []
        m = multiprocessing.Array('i',n*n)
        for i in range(workers_count):
            process = multiprocessing.Process(target=minor_worker, args=(A,m,i,workers_count))
            process.start()
            processes.append(process)
        
        for p in processes:
            p.join()
        
      
        for k,v in enumerate(m[:]):
            i = k//n
            j = k % n
            B[i,j]=v
        return B
    
    
    for i in range(0,n):
        for j in range(0,n):
            B[i,j]=A.minor(i,j)
    return B
    
def minor_worker(A,m,offset,count):
    n=A.rows

    for k in range(offset,n*n,count):
        i = k//n
        j = k % n
        m[k] = A.minor(i,j)


    return 0       



def get_base(n,c):
    p = math.log(n)*math.log(math.log(n))
    p = math.sqrt(p) / 2
    limit = c* math.exp(p)
    base = []
    for p in primes:
        if p > limit:
            break
        base.append(p)
    return base

def get_equations(base,a,n):
    matrix,vector = [], []
    rank = -1
    while rank != len(base):
        k = random.randint(0,n-1)
        num = pow(a,k,n)
        if num == 1: continue
        v = factorize(base,num)
        if v != None:
            new_rank = np.linalg.matrix_rank([*matrix,v],)
            if(new_rank > rank):
                matrix.append(v)
                vector.append(k)
                rank = new_rank       
    return matrix,vector
    
def factorize(base,value): 
    vector = [0]*len(base)
    for i,p in enumerate(base):
        while value != 1:
            if value % p == 0:
                vector[i] += 1
                value = value // p
            else:
                break

    return vector if value == 1 else None

def inverse_matrix(inv_matrix,vector,n):
    return (inv_matrix * vector).applyfunc(lambda x: x % (n-1))

def find_log(a,b,n,base,vector):
    while True:
        l = random.randint(0,n-2)
        num = (b*pow(a,l,n))%n
        factors = factorize(base,num)
        if factors != None: break

    answer = -l
    for i in range(len(base)):
        answer += factors[i]*vector[i] 

    return int(answer  % (n-1))    

