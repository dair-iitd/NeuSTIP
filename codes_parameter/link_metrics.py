import pickle
import torch
import sys
import os

char_string = sys.argv[1]
num_rel = int(sys.argv[2])
metrics = {"mr", "mrr", "h@1","h@3","h@10"}
inference = {"valid","test"}
net_metrics = dict()
for inf in inference:
    net_metrics[inf] = dict()
    for m in metrics:
        net_metrics[inf][m] = 0.0
        net_metrics[inf]['size'] = 0
        

for r in range(num_rel):
    for inf in inference:
        input_file = f'./workspace/metrics/r{r}/{inf}_{char_string}'
         
        if ((os.path.exists(input_file))):
            with open(input_file, 'rb') as pickle_file:
                input_dict = pickle.load(pickle_file)
                input_size = input_dict[inf+'_data_size']
               

                net_metrics[inf]['size'] += input_size
               

                for m in metrics:
                    net_metrics[inf][m] += input_size*input_dict[m]
                   
for inf in inference:
    print(f"-- {inf} stats --")
    print('-- all --')
    for m in metrics:
        print(m+" : {}".format(net_metrics[inf][m]/net_metrics[inf]['size']))


