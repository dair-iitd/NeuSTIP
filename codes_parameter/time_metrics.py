import pickle
import torch
import sys
import os

char_string = sys.argv[1]
num_rel = int(sys.argv[2])
dataset = sys.argv[3]
score_func = {"gIOU", "aeIOU", "TAC", "IOU", "gaeIOU"}
iouatk = [1]
inference = {"valid","test"}
relation_classes = {"all","instant","short","long"}
rel_wise_scores = dict()
class_type = dict()
for cl in relation_classes:
  rel_wise_scores[cl] = dict()
  for inf in inference:
    rel_wise_scores[cl][inf] = dict()
    for score_func1 in score_func:
      rel_wise_scores[cl][inf][score_func1] = list()

with open(f'../data/{dataset}/class_info.txt','r') as f_class:
# See performance across relation classes as well
  for line in f_class.readlines():
    idd, class_r = line.split()
    class_type[int(idd)] = class_r

for r in range(num_rel):
  for inf in inference:
    input_file = './workspace/aeIOU/r'+str(r)+'/'+char_string+f'_{inf}_time_score_analysis'
    if ((os.path.exists(input_file))):

      with open(input_file, 'rb') as pickle_file:
        mydict = pickle.load(pickle_file)
        if (inf == 'test'):
          print(torch.mean(mydict['scores_dict'][(1,"aeIOU")]))
        for score_func1 in score_func:
          rel_wise_scores["all"][inf][score_func1].append(mydict['scores_dict'][(1,score_func1)])
          rel_wise_scores[class_type[r]][inf][score_func1].append(mydict['scores_dict'][(1,score_func1)])

for cl in relation_classes:
  print(f"-- relation class: {cl} --")
  for inf in inference:
    print(f"-- {inf} stats --")
    for score_func1 in score_func:
      if (len(rel_wise_scores[cl][inf][score_func1])!=0):
        print(score_func1+" : {}".format(torch.mean(torch.cat(rel_wise_scores[cl][inf][score_func1],dim=0))))
      else:
        print("empty list")