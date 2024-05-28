from time import time
from index_calc import index_calc


a = 76200214
b = 148042480
n = 150152773


if __name__ == '__main__':                
    start = time()
    log = index_calc(a,b,n,True, 8)
    print('[*] ASync index-calculus')
    print('     time:',time()-start)
    print('     result:',log)

    start = time()
    log = index_calc(a,b,n)
    print('[*] Sync index-calculus')
    print('     time:',time()-start)
    print('     result:',log)


