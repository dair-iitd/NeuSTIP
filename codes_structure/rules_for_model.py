# For model
import sys
input_file = sys.argv[1]
output_rule_directory = sys.argv[2]
dataset_name = sys.argv[3]
num_relations = int(sys.argv[4])
output_file_suffix = ''
fin = open(input_file,"r")
rel_wise_rules = [[] for i in range(num_relations)]
act_rel_to_id= dict()
idto_rel_str = dict()
f1 = open('../data/'+dataset_name+'/relations.dict','r')
for l in f1.readlines():
	idd,rel = l.split()
	act_rel_to_id[int(rel)] = int(idd)
f1.close()
f2 = open('../data/'+dataset_name+'/relation2id.txt','r')
for l in f2.readlines():
	rel, idd = l.split('\t')
	idto_rel_str[act_rel_to_id[int(idd)]] = rel
f2.close()

for line in fin.readlines():
	rule_head = int(line.split()[0])
	rule_body = line.split()[1:-2]
	pca_conf = float(line.split()[-2])
	grnd = int(line.split()[-1])
	rel_wise_rules[rule_head].append((pca_conf,rule_body,grnd))

rules_per_rel = [0 for i in range(num_relations)]
total_cnt = 0
for i in range(num_relations):
	rel_wise_rules[i].sort(reverse=True)
	fout = open('../data/'+dataset_name+'/'+output_rule_directory+"/rules_"+str(i)+output_file_suffix+".txt","w")
	for rules in rel_wise_rules[i]:
		rules_per_rel[i] +=1
		total_cnt +=1
		rule_body = rules[1]
		for j in range(int(len(rule_body))):
			fout.write(rule_body[j])
			if (j!=int(len(rule_body))-1):
			    fout.write(' ')
		fout.write('\t'+str(rules[0])+'\t'+str(rules[2])) 
		fout.write('\n')

for i in range(num_relations):
	print(f"Total rules for relation {idto_rel_str[i]} =  {rules_per_rel[i]}")
