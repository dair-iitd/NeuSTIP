import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
from easydict import EasyDict
import numpy as np
import torch
import knowledge_graph_utils
from knowledge_graph_utils import *

def allen_relation(t1, t2, cur_t1, cur_t2, atemporal): 
        
        if (atemporal == 1):
            return 13 

        if (t2 < cur_t1): # Before 
            return 0
        if (cur_t2 < t1): # After 
            return 1

        if (clubbed == 1): 
            return 4

        if (t1 == cur_t1 and t2 == cur_t2): #EQUALS
            return 12

        if (t2 == cur_t1): # Meets 
            return 2
        if (cur_t2 == t1): # !Meets
            return 3
        if (t2 > cur_t1 and t2 < cur_t2 and t1 < cur_t1): #Overlaps
            return 4
        if (cur_t2 > t1 and cur_t2 < t2 and cur_t1 < t1): #!overlaps
            return 5
        if (t1 == cur_t1 and t2 < cur_t2): # starts
            return 6
        if (cur_t1 == t1 and cur_t2 < t2): # !starts
            return 7
        if (t1 > cur_t1 and t2 < cur_t2): # during
            return 8
        if (cur_t1 > t1 and cur_t2 < t2): # !during
            return 9
        if (t1 > cur_t1 and t2 == cur_t2): # finishes
            return 10
        if (cur_t1 > t1 and t2 == cur_t2): # finishes
            return 11


def walk(graph, hrT2o, start, node, depth, NUM_RELATIONS):
	# start is h,t1,t2, r,t 
    if (depth == DEPTHS - 1):
        rules = []
        for relation in range(NUM_RELATIONS):
            candidates = list(graph[node[0]][relation])
            for candidate in candidates:
                head_relation = start[3]
                if (candidate[0] == start[4]): 
                    if not(((start[0],head_relation,start[1],start[2],candidate[0])==(node[0],relation,candidate[1],candidate[2],candidate[0]))):
                       

                        rules.append([head_relation, [allen_relation(node[1],node[2],candidate[1],candidate[2],atemporal)*NUM_RELATIONS + relation], candidate[0], [(node[0],candidate[0],candidate[1],candidate[2])]]) # can be reached from rule head also
	                    
        if (len(rules) == 0):
            return False, []
        else:
            return True, rules
    else:
        new_rules = []
        for relation in range(NUM_RELATIONS):
            candidates = list(graph[node[0]][relation])
            for candidate in candidates:
                found, rules = walk(graph, hrT2o, start, candidate, depth+1, NUM_RELATIONS) 
               
                for i in range(len(rules)):
                    if (((start[0],rules[i][0],start[1],start[2],rules[i][2]) == (node[0],relation,candidate[1],candidate[2],candidate[0]))):
                 
                        continue
        
            
                    rules[i][1] = [allen_relation(node[1],node[2],candidate[1],candidate[2],atemporal)*NUM_RELATIONS + relation] + rules[i][1] # Also add allen_relation before [relation] determined by node's t1,t2
                    rules[i][3] = [(node[0],candidate[0],candidate[1],candidate[2])] + rules[i][3]
                        
                    new_rules.append(rules[i])

                del rules
                head_relation = start[3]
                if (candidate[0] == start[4]):


                    if not((start[0],head_relation,start[1],start[2],candidate[0])==(node[0],relation,candidate[1],candidate[2],candidate[0])):
                        new_rules.append([head_relation, [allen_relation(node[1],node[2],candidate[1],candidate[2],atemporal)*NUM_RELATIONS + relation], candidate[0], [(node[0],candidate[0],candidate[1],candidate[2])]])



                    else:
                        if (depth ==0): # Only do that for depth = 0, to avoid r equals r rule
                            if not((start[0],head_relation,start[1],start[2],candidate[0])==(node[0],relation,candidate[1],candidate[2],candidate[0])):
                                new_rules.append([head_relation, [allen_relation(node[1],node[2],candidate[1],candidate[2],atemporal)*NUM_RELATIONS + relation], candidate[0], [(node[0],candidate[0],candidate[1],candidate[2])]])
                        else:
                            new_rules.append([head_relation, [allen_relation(node[1],node[2],candidate[1],candidate[2],atemporal)*NUM_RELATIONS + relation], candidate[0], [(node[0],candidate[0],candidate[1],candidate[2])]])
            

        if (len(new_rules) > 0):
        	return True, new_rules
        else:
        	return False, []


def perform_walks(graph, hT_list, hT_list_per_rel, hrT2o, entity_size, DATASET, NUM_RELATIONS, NUM_WALKS, OUT_FILE, DEVICE):
    rules_total = set()
    num_grounded = 0
    for elements in hT_list:
        if (elements[3] != -1):
            for _ in range(NUM_WALKS):
                found, rules = walk(graph,hrT2o, elements, elements, 0, NUM_RELATIONS)
                if (found):
                    for rule in rules:
                        rules_total.add(tuple([rule[0], tuple(rule[1])]))
   
    print(f'Rules Discovered: {len(rules_total)}')
    rules_total = list(rules_total)
    rules_total.sort()

    final_rules = set()
    for rule in rules_total:
        final_rules.add((rule,0,0))

    with open(OUT_FILE, 'w') as fi:
        for rule in final_rules:
            fi.write(str(rule[0][0]) + " " + " ".join([str(_) for _ in list(rule[0][1])]) +" "+str(rule[1])+ " "+str(rule[2])+'\n')

DATASET = sys.argv[1]
OUT_FILE = sys.argv[2]
dataset_ratio = float(sys.argv[3]) # a floating point number between 0-1, denoting the fraction of the training data which we require
DEPTHS = int(sys.argv[4]) 
atemporal = int(sys.argv[5]) # 1 denotes we want to learn Non-temporal rules
entity_inductive = int(sys.argv[6]) # 1, if using entity inductive setting
clubbed = int(sys.argv[7]) # Experiment to Club the Allen predicates together

if (DATASET == 'YAGO11k'):
    NUM_RELATIONS = 20

elif (DATASET == 'WIKIDATA12k'):
    NUM_RELATIONS = 48

else:
    sys.exit('Invalid Dataset!')

DATA = f'../data/{DATASET}'

dataset = load_dataset_temporal('../data/'+DATASET,DATASET,0, entity_inductive,dataset_ratio)
graph = build_graph(dataset['train'],dataset['E'],dataset['R']) 

hrT2o = dict()
hT_list = set()
hT_list_per_rel = [set() for i in range(NUM_RELATIONS)]
for h,r,t,t1,t2 in dataset['train']: 
    if ((h,r,t1,t2) not in hrT2o.keys()):
        hrT2o[(h,r,t1,t2)] = set()

    hrT2o[(h,r,t1,t2)].add(t)
    hT_list_per_rel[r].add((h,t1,t2))
    hT_list.add((h,t1,t2,r,t))
    
perform_walks(graph, hT_list, hT_list_per_rel, hrT2o,dataset['E'],DATASET, NUM_RELATIONS,1, OUT_FILE, 'cpu')
