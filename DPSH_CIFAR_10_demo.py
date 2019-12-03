import DPSH_CIFAR_10 as dpsh
import pickle
from datetime import datetime
import argparse

def DPSH_CIFAR_10_demo(opt):
    param = {}
    param['lambda'] = opt.lamda
    bits = [12, 24, 32, 48]
    for bit in bits:
        filename = 'log/DPSH_' + str(bit) + 'bits_CIFAR-10' + '_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
        param['filename'] = filename
        print('---------------------------------------')
        print('[#bit: %3d]' % (bit))
        result = dpsh.DPSH_algo(bit, param, opt.gpu)
        print('[MAP: %3.5f]' % (result['map']))
        print('---------------------------------------')
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default = 0, type = int, help = 'gpu no')
    parser.add_argument('--lamda', default = 10, type = int, help = 'hyperparam lambda')
    opt = parser.parse_args()
    DPSH_CIFAR_10_demo(opt)
