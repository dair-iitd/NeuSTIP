import sys
import os
import argparse
from knowledge_graph_utils import *
from kb import *
from main import NeuSTIP


DATA_name = sys.argv[1]
DATA_DIR = '../data/'+DATA_name
OUTPUT_DIR  = './workspace'
eta = float(sys.argv[2]) # Weight of TimePlex
start_rel = int(sys.argv[3])
end_rel = int(sys.argv[4])
lr = 1e-3
pe = -1
atemporal = int(sys.argv[5]) # For comparison with atemporal rules, for Link Prediction
mode = sys.argv[6] # Can be 'link' or 'time'
use_duration = int(sys.argv[7]) # 1 if using duration distribution in time interval prediction
max_depth = int(sys.argv[8]) # max rule length
entity_inductive = int(sys.argv[9]) # 1 if generalising to unseen entities
dataset_ratio = float(sys.argv[10]) # Fraction of training data to be used for limited setting experiment
clb = int(sys.argv[11]) # 1 if using clubbed predicates
if (mode == 'link'):
    pe = 5000
else:
    pe = 2000

temp_str = '_temp'
use_duration_str = ''
entity_inductive_str = ''
ratio_str = ''
clb_str = ''
if (atemporal == 1):
    temp_str = '_atemp'
if (use_duration == 1):
	use_duration_str = '_use_duration'

if (entity_inductive == 1):
    entity_inductive_str = '_entity_ind'
if (dataset_ratio < 1):
    ratio_str = f'_{dataset_ratio}'
if (clb== 1):
    clb_str = '_clb'


rule_file_str = f'Rules_{max_depth}'+temp_str+clb_str+entity_inductive_str+ratio_str
parameter_str = DATA_name+"_"+str(eta)+'_'+str(lr)+'_'+str(pe)+temp_str+clb_str+use_duration_str+'_'+str(max_depth)+entity_inductive_str+ratio_str
old_print = print
predict_time = 0
if (mode == 'time'):
	predict_time = 1
dataset = load_dataset_temporal(f"{DATA_DIR}", DATA_name, predict_time, entity_inductive, dataset_ratio) 
log_filename = f"common.txt"
log_file = open(log_filename, 'w')



def new_print(*args, **kwargs):
    old_print(*args, **kwargs, flush=True)
    old_print(*args, **kwargs, file=log_file, flush=True)

print = new_print

kb_dict = dict()


use_time_tokenizer = False
introduce_oov=1 
datamap   = Datamap(DATA_DIR.split("/")[-1],DATA_DIR, False)
ktest_sub =  kb(datamap, os.path.join(DATA_DIR, 'intervals/test.txt'), add_unknowns=int(not (int(introduce_oov))), use_time_tokenizer=use_time_tokenizer)
kvalid_sub = kb(datamap, os.path.join(DATA_DIR, 'intervals/valid.txt'), add_unknowns=int(not (int(introduce_oov))), use_time_tokenizer=use_time_tokenizer)
kb_dict['test'] = ktest_sub
kb_dict['valid'] = kvalid_sub

model = NeuSTIP(dataset, kb_dict, eta, lr, pe, use_duration, max_depth, clb, print=print)


start = start_rel
end = end_rel

for r in range(start,end):
   
    rule_file = f"{DATA_DIR}/"+rule_file_str+f"/rules_{r}.txt"
    metrics_filename = parameter_str
    model.train_model_start(r,rule_file=rule_file,model_file=f"{OUTPUT_DIR}/model_" + parameter_str + f"_{r}.pth", threshold_file = metrics_filename, train_mode = mode ) # See use of Rule file here
