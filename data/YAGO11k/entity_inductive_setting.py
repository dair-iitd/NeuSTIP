# Randomly select a subset of test quadruples to be the new test set
# From the train set (used for rule learning and confidence estimation, remove common entities from the new test set)
# Learn the rules on the new train set, also ground them for confidence estimation on the new train set

import sys
import random
import numpy as np
random.seed(0)
fraction = float(sys.argv[1])
predict_time = int(sys.argv[2])
f_train = open('train.txt','r')
if (predict_time == 1):
	f_test = open('./intervals/test.txt','r')
	f_valid = open('./intervals/valid.txt','r')
	f_valid_new = open('valid_ent_ind_time.txt','w')
	f_test_new = open('test_ent_ind_time.txt','w')
	f_train_new = open('train_ent_ind_time.txt','w')
else:

	f_test = open('test.txt','r')
	f_valid = open('valid.txt','r')
	f_valid_new = open('valid_ent_ind_link.txt','w')
	f_test_new = open('test_ent_ind_link.txt','w')
	f_train_new = open('train_ent_ind_link.txt','w')

num_rel = int(sys.argv[3])
test_data = []
train_data = []
test_size = 0
for line in f_test.readlines():
	h,r,t,t1,t2 = line.split('\t')
	if (int(r) < num_rel/2):
		test_data.append(np.array([int(h),int(r),int(t),int(t1),int(t2)]))
		test_size +=1

for line in f_train.readlines():
	h,r,t,t1,t2 = line.split('\t')
	if (int(r) < num_rel/2):
		train_data.append(np.array([int(h),int(r),int(t),int(t1),int(t2)]))

for line in f_valid.readlines():
	f_valid_new.write(line)
f_valid_new.close()

new_test_size = int(fraction*test_size)

test_new = random.sample(test_data,new_test_size)
entity_set = set()
for i in range(len(test_new)):
	entity_set.add(test_new[i][0]) # h
	entity_set.add(test_new[i][2]) # t

	f_test_new.write(str(test_new[i][0])+'\t'+str(test_new[i][1])+'\t'+str(test_new[i][2])+'\t'+str(test_new[i][3])+'\t'+str(test_new[i][4])+'\n')
	f_test_new.write(str(test_new[i][2])+'\t'+str(int(test_new[i][1]+num_rel/2))+'\t'+str(test_new[i][0])+'\t'+str(test_new[i][3])+'\t'+str(test_new[i][4])+'\n')
f_test_new.close()

new_train_size = 0
for i in range(len(train_data)):
	h = train_data[i][0]
	t = train_data[i][2]
	if ((h not in entity_set) and (t not in entity_set)):
		f_train_new.write(str(train_data[i][0])+'\t'+str(train_data[i][1])+'\t'+str(train_data[i][2])+'\t'+str(train_data[i][3])+'\t'+str(train_data[i][4])+'\n')
		f_train_new.write(str(train_data[i][2])+'\t'+str(int(train_data[i][1]+num_rel/2))+'\t'+str(train_data[i][0])+'\t'+str(train_data[i][3])+'\t'+str(train_data[i][4])+'\n')
		new_train_size +=2
f_train_new.close()
print(new_train_size)


