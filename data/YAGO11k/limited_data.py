import sys
import random
random.seed(0)
ratio = float(sys.argv[1])
num_rel = int(sys.argv[2])
f = open('train.txt','r')
f_out = open('train_'+str(ratio)+'.txt','w')
list_ = []
for line in f.readlines():
	h,r,t,t1,t2 = line.split('\t')
	if (int(r) < num_rel/2):
		list_.append([int(h),int(r),int(t),int(t1),int(t2)])
random.shuffle(list_)

list_ = random.sample(list_,int(len(list_)*ratio))
for line in list_:
	f_out.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\t'+str(line[4])+'\n')
	f_out.write(str(line[2])+'\t'+str(int(line[1] + num_rel/2))+'\t'+str(line[0])+'\t'+str(line[3])+'\t'+str(line[4])+'\n')
f_out.close()