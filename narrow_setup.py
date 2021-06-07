
# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time
 
 
def gpu_info(gpu_id=0):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[gpu_id * 4 + 2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[gpu_id * 4 + 1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory
 
 
def narrow_setup(gpu_id, gpu_id2, cmd, interval=2):
    _, gpu_memory1 = gpu_info(gpu_id)
    _, gpu_memory2 = gpu_info(gpu_id2)
    i = 0
    while gpu_memory1 > 1000 or gpu_memory2 > 1000:  # set waiting condition
        _, gpu_memory1 = gpu_info(gpu_id)
        _, gpu_memory2 = gpu_info(gpu_id2)
        i = i % 10
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_memory_str1 = 'gpu memory1:%d MiB |' % gpu_memory1
        gpu_memory_str2 = 'gpu memory2:%d MiB |' % gpu_memory2
        sys.stdout.write('\r' + gpu_memory_str1 + ' ' + gpu_memory_str2 + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)
 
 
if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    gpu_id2 = int(sys.argv[2])
    cmd = 'sh ' + sys.argv[3]
    narrow_setup(gpu_id, gpu_id2, cmd)
