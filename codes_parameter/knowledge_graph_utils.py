import torch
import os
import numpy 
import collections
from collections import defaultdict

YEARMIN = 0  # -50
YEARMAX = 3000

YEAR_STR_LEN = 4 

def get_pairwise_r_dict(ent_rel_dict_t1, ent_rel_dict_t2):
    """
    Given a dict - dic[entity][relation][time], which contains details of all relations
    and time a given set of entities (when in sub/obj position) are seen with
    Returns:
    rxr dict: containg list of all time differences of r1 and r1
    """
    r_r_dict = {}
    for entity in ent_rel_dict_t1:
        for r1 in ent_rel_dict_t1[entity]:
            for r2 in ent_rel_dict_t2[entity]:
                for r1_t in list(ent_rel_dict_t1[entity][r1]):  # t1 time for r1 (can be start/end)
                    for r2_t in list(ent_rel_dict_t2[entity][r2]):  # t2 time for r2 (can be start/end)
                        if not r1 in r_r_dict.keys():
                            r_r_dict[r1] = {}
                        if not r2 in r_r_dict[r1].keys():
                            r_r_dict[r1][r2] = []
                       
                        r_r_dict[r1][r2].append(int(r1_t) - int(r2_t))

    return r_r_dict

def load_dataset_temporal(DATA_DIR,name,predict_time,entity_inductive, dataset_ratio):
   
    entity2id = dict()
    relation2id = dict()
    time2id = dict()
    id2time = dict()


    with open(f'{DATA_DIR}/entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(f'{DATA_DIR}/relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    with open(f'{DATA_DIR}/time.dict') as fin:
        time2id = dict()
        for line in fin:
            tid, time = line.strip().split('\t')
            time2id[time] = int(tid)
            id2time[int(tid)] = time

    E = len(entity2id)
    R = len(relation2id)
    T = len(time2id)

    ret = dict()
    ret['E'] = E
    ret['R'] = R
    ret['T'] = T
    ret['id2time'] = id2time
    ret['name'] = name

    knowns = dict()
    item_base = ['train','valid','test']
    train_base_file = 'train'

    if (dataset_ratio < 1):
        train_base_file = f'train_{dataset_ratio}'

    filename = [train_base_file,'valid','test']

    background_kg = [train_base_file,'valid']

    if predict_time == 1:
        filename = [train_base_file,'intervals/valid','intervals/test']
        background_kg = [train_base_file,'intervals/valid']

    if (entity_inductive == 1):
        if (predict_time == 0):
            filename = ['train_ent_ind_link','valid_ent_ind_link','test_ent_ind_link']
            background_kg = ['train','valid_ent_ind_link']
        else:
            filename = ['train_ent_ind_time','intervals/valid_ent_ind_time','intervals/test_ent_ind_time']
            background_kg = ['train','intervals/valid_ent_ind_time']

    rel_dict_t1_start = defaultdict(lambda: defaultdict(set))
    rel_dict_t2_start = defaultdict(lambda: defaultdict(set))
    rel_dict_t1_end = defaultdict(lambda: defaultdict(set)) # For end-end distribution
    rel_dict_t2_end = defaultdict(lambda: defaultdict(set)) # For end-end distribution
    mean_ilength = torch.zeros(R)
    var_ilength = torch.zeros(R)
    ilengths_data = defaultdict(list) # dictionary of lists
    no_grnd_start_year = torch.zeros(R)
    start_id_data = defaultdict(list)
    no_grnd_end_year = torch.zeros(R)
    offset_id_data = defaultdict(list)
    test_sizes = torch.zeros(R)
    recurrent_dict_start = defaultdict(lambda: defaultdict(list))
    recurrent_dict_end = defaultdict(lambda: defaultdict(list))

    for i in range(len(filename)):
        item = filename[i]
        edges = []
        with open(f"{DATA_DIR}/{item}.txt") as fin:
            for line in fin:
                h, r, t, t1, t2 = line.strip().split('\t')
  
                h, r, t, t1, t2 = entity2id.get(h,len(entity2id)-1), relation2id[r], entity2id.get(t,len(entity2id)-1), time2id[t1], time2id[t2] 
                if int(t1) > int(t2):
                    t2 = t1

                if (item_base[i] == 'train'):

                    if (int(id2time[t1])!=YEARMIN):
                        start_id_data[r].append(int(t1))
                    
                        rel_dict_t1_start[h][r].add(id2time[t1])
                        rel_dict_t2_start[h][r].add(id2time[t1])

                        recurrent_dict_start[r][h].append(int(id2time[t1]))

                        if (int(id2time[t2])!=YEARMAX): # avoid missing cases
                            ilengths_data[r].append(int(id2time[t2])-int(id2time[t1]) + 1)
                            offset_id_data[r].append(int(t2) - int(t1))
                            rel_dict_t1_end[h][r].add(id2time[t2])
                            rel_dict_t2_end[h][r].add(id2time[t2])

                            recurrent_dict_end[r][h].append(int(id2time[t2]))

                edges.append([h, r, t, t1, t2])

                for time in range(t1,t2+1):
                    if ((h,r,time) not in knowns):
                        knowns[(h,r,time)] = set()
                    knowns[(h,r,time)].add(t)
                    
            ret[item_base[i]] = edges

    edges_bg = []
    for i in range(len(background_kg)):
        item = background_kg[i]
        with open(f"{DATA_DIR}/{item}.txt") as fin:
            for line in fin:
                h, r, t, t1, t2 = line.strip().split('\t')
                h, r, t, t1, t2 = entity2id.get(h,len(entity2id)-1), relation2id[r], entity2id.get(t,len(entity2id)-1), time2id[t1], time2id[t2] 
                
                if int(t1) > int(t2):
                    t2 = t1

                edges_bg.append([h, r, t, t1, t2])
                
    ret['bg'] = edges_bg
        

    r_r_dict_start = get_pairwise_r_dict(rel_dict_t1_start, rel_dict_t2_start)
    r_r_dict_end = get_pairwise_r_dict(rel_dict_t1_end, rel_dict_t2_end)
    r_diffs_start = defaultdict(list)
    r_diffs_end = defaultdict(list)

    for r in recurrent_dict_start:
        for h in recurrent_dict_start[r]:
            time_list = sorted(recurrent_dict_start[r][h])
          
            if (len(time_list) > 1):
                for idx,time in enumerate(time_list[1:]):

                    r_diffs_start[r].append(time - time_list[idx])

    for r in recurrent_dict_end:
        for h in recurrent_dict_end[r]:
            time_list = sorted(recurrent_dict_end[r][h])
          
            if (len(time_list) > 1):
                for idx,time in enumerate(time_list[1:]):

                    r_diffs_end[r].append(time - time_list[idx])

    r_stat_start = {}
    for r in r_diffs_start:
        data = r_diffs_start[r]
        r_stat_start[r] = (numpy.mean(data),numpy.var(data),len(data))
       
    r_stat_end = {}
    for r in r_diffs_end:
        data = r_diffs_end[r]
        r_stat_end[r] = (numpy.mean(data),numpy.var(data),len(data))

    mean_r_start = torch.zeros(R)
    mean_r_end = torch.zeros(R)
    var_r_start = torch.zeros(R)
    var_r_end = torch.zeros(R)

    r_r_stat_start = {}
    r_r_stat_end = {}

    for r1 in r_r_dict_start.keys():
        for r2 in r_r_dict_start[r1].keys():
            data = r_r_dict_start[r1][r2]
            r_r_stat_start[r1, r2] = (numpy.mean(data), numpy.var(data), len(data))

    for r1 in r_r_dict_end.keys():
        for r2 in r_r_dict_end[r1].keys():
            data = r_r_dict_end[r1][r2]
            r_r_stat_end[r1, r2] = (numpy.mean(data), numpy.var(data), len(data))

    inf = 1000
    min_var = 0.01
    min_support = 10


    mean_r_r_start = torch.zeros(R, R)
    var_r_r_start = torch.zeros(R,R)

    mean_r_r_end = torch.zeros(R, R)
    var_r_r_end = torch.zeros(R,R)

    for i in range(R):
    
        mean_s, var_s, sup_s = r_stat_start.get(i,(-inf,0.01,0))
        var_s = max(min_var,var_s)

        mean_e, var_e, sup_e = r_stat_end.get(i,(-inf,0.01,0))
        var_e = max(min_var,var_e)

        mean_r_start[i] = mean_s
        var_r_start[i] = var_s

        mean_r_end[i] = mean_e
        var_r_end[i] = var_e

        mean_ilength[i] = numpy.mean(ilengths_data[i])
        var_ilength[i] = numpy.var(ilengths_data[i]) + min_var # adding epsilon term, to avoid 0 var case
        if (len(ilengths_data[i]) == 0):
            mean_ilength[i] = 0
            var_ilength[i] = min_var
        if (len(start_id_data[i])==0 or len(offset_id_data[i]) == 0):
            no_grnd_start_year[i] = 0
            no_grnd_end_year[i] = 0
        else:
            no_grnd_start_year[i] = numpy.mean(start_id_data[i]) # can be non integers
            no_grnd_end_year[i] = numpy.mean(offset_id_data[i]) + no_grnd_start_year[i] # can be non integers

    for i in range(R):
        for j in range(R):
            mean, var, sup = r_r_stat_start.get((i, j), (-inf, min_var, 0))
            mean1, var1, sup1 = r_r_stat_end.get((i, j), (-inf, min_var, 0))

            var = max(var, min_var)
            var1 = max(var1, min_var)
           
            mean_r_r_start[i, j] = mean
            var_r_r_start[i, j] = var

            mean_r_r_end[i, j] = mean1
            var_r_r_end[i, j] = var1
 

    torch.save(mean_r_r_start,f'mean_r_r_start_{name}')
    torch.save(var_r_r_start,f'var_r_r_start_{name}')

    torch.save(mean_r_r_end,f'mean_r_r_end_{name}')
    torch.save(var_r_r_end,f'var_r_r_end_{name}')
    torch.save(mean_ilength,f'mean_ilength_{name}')
    torch.save(var_ilength,f'var_ilength_{name}')
    torch.save(no_grnd_start_year,f'no_grnd_start_year_{name}')
    torch.save(no_grnd_end_year,f'no_grnd_end_year_{name}')
    torch.save(mean_r_start,f'mean_r_start_{name}')
    torch.save(mean_r_end,f'mean_r_end_{name}')
    torch.save(var_r_start,f'var_r_start_{name}')
    torch.save(var_r_end,f'var_r_end_{name}')


            
    for k in knowns: # LIST Conversion
            knowns[k] = list(knowns[k])
    ret['knowns'] = knowns

    ret['mean_r_r_start'] = torch.load(f'mean_r_r_start_{name}')
    ret['var_r_r_start'] = torch.load(f'var_r_r_start_{name}')
    ret['mean_r_r_end'] = torch.load(f'mean_r_r_end_{name}')
    ret['var_r_r_end'] = torch.load(f'var_r_r_end_{name}')

    ret['mean_r_start'] = torch.load(f'mean_r_start_{name}')
    ret['var_r_start'] = torch.load(f'var_r_start_{name}')
    ret['mean_r_end'] = torch.load(f'mean_r_end_{name}')
    ret['var_r_end'] = torch.load(f'var_r_end_{name}')

    ret['mean_ilength'] = torch.load(f'mean_ilength_{name}')
    ret['var_ilength'] = torch.load(f'var_ilength_{name}')
    ret['no_grnd_start_year'] = torch.load(f'no_grnd_start_year_{name}')
    ret['no_grnd_end_year'] = torch.load(f'no_grnd_end_year_{name}')


    return ret

def build_graph(edges, E, R):
    # returns a dict for [entity][relation] to t,t1,t2
    e2r2 = [[set() for k in range(R)] for i in range(E)]
    inve2r2 = [[set() for k in range(R)] for i in range(E)]
    for edge in edges:
        h, r, t, t1, t2 = edge
   
        e2r2[h][r].add((t,t1,t2)) # to avoid duplicates
        inve2r2[t][r].add((h,t1,t2)) # to avoid duplicates
    for i in range(E):
        for j in range(R):
            e2r2[i][j] = list(e2r2[i][j])
            inve2r2[i][j] = list(inve2r2[i][j])
    return e2r2


def dataset_graph(dataset, edges='train'):
    return Graph(dataset[edges], num_node=dataset['E'], num_relation=dataset['R'], num_times = dataset['T'])


def list2mask(a, N):
    if isinstance(a, list):
        a = torch.LongTensor(a)
    m = torch.zeros(N).to(a.device).bool()
    m[a] = True
    return m


def mask2list(m):
    N = m.size(0)
    m = m.cuda()
    a = torch.arange(N).cuda()
    return a[m]



class Datamap(object):
    """
    Creates and stores entity/relation/time maps for a given dataset
    """
    def __init__(self, dataset, dataset_root, use_time_interval=False):
        self.dataset      = dataset
        self.dataset_root = dataset_root
        self.use_time_interval=use_time_interval

        self.unk_time_str="UNK-TIME"

        train_filename=os.path.join(dataset_root, 'train.txt')


        # ---entity/relation maps--- #
        self.entity_map = {}
        self.relation_map = {}
        self.reverse_entity_map = {}
        self.reverse_relation_map = {}
        
        time2id = dict()
        id2time = dict()

        with open(f'{dataset_root}/entities.dict') as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                self.entity_map[entity] = int(eid)
                self.reverse_entity_map[int(eid)] = entity

        with open(f'{dataset_root}/relations.dict') as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                self.relation_map[relation] = int(rid)
                self.reverse_relation_map[int(rid)] = relation
                 
        with open(f'{dataset_root}/time.dict') as fin:
            time2id = dict()
            for line in fin:
                tid, time = line.strip().split('\t')
                if time in self.unk_time_str:
                    time2id[self.unk_time_str] = tid
                    id2time[int(tid)] = self.unk_time_str
                else:
                    time2id[int(time)] = int(tid)
                    id2time[int(tid)] = int(time)
        
        self.dateYear2id = time2id
        self.id2dateYear = id2time
        
        #print("self.dateYear2id :{}".format(self.dateYear2id))
        
        # ---time maps--- #
        self.dateYears2intervalId, self.intervalId2dateYears, self.timeStr2Id, self.id2TimeStr = self.get_time_info(dataset_root)

        self.year2id = {} # needed if use_time_interval is True
        with open(train_filename) as f:
            lines = f.readlines()
            lines = [l.strip("\n").split("\t") for l in lines]
            # ----Mapping of time-interval-tuple to id----#
            if self.use_time_interval and (len(self.year2id) == 0):
                triple_time = dict()
                count = 0
                for line in lines:
                    triple_time[count] = [x.split('-')[0] for x in line[3:5]]
                    count += 1
                self.year2id = self.create_year2id(triple_time, bin_size=300)  # (bin_start, bin_end) to id
            # ------- #

        # time maps converted to a form that can be indexed
        self.id2dateYear_mat = self.convert_dict2mat(self.dateYear2id)
        self.intervalId2dateYears_mat_s, self.intervalId2dateYears_mat_e = self.convert_dict2mat_tup(self.dateYears2intervalId)
        self.intervalId2dateYearsId_mat_s = self.convert_year2id(self.intervalId2dateYears_mat_s, self.dateYear2id)
        self.intervalId2dateYearsId_mat_e = self.convert_year2id(self.intervalId2dateYears_mat_e, self.dateYear2id)
        self.binId2year_mat = self.convert_dict2mat(self.year2id)
        

    def get_time_info(self, DATA_DIR="", predict_time=True):
        '''
        Reads all data (train+test+valid) and returns date(year) to id and time interval to id maps
        including their inverse maps
        '''
        #files_to_read = ['train', 'test', 'valid']

        if predict_time:
            files_to_read = ['train', 'intervals/valid', 'intervals/test']
        else:
            files_to_read = ['train', 'valid', 'test']

        all_years = []
        all_intervals = []
        dateYear2id = {}
        id2dateYear = {}
        dateYears2intervalId = {}
        intervalId2dateYears = {}
        timeStr2Id = {}
        id2TimeStr = {}
        time_str=self.unk_time_str
        for filename in files_to_read:
            #with open(os.path.join(dataset_root, filename)) as f:
            with open(f"{DATA_DIR}/{filename}.txt") as f:
                lines = f.readlines()
                lines = [l.strip("\n").split("\t") for l in lines]
                for l in lines:
                    if len(l) == 5:
                        date1 = self.check_date_validity(l[3])
                        if date1 != -1:
                            all_years.append(date1)

                        date2 = self.check_date_validity(l[4])
                        if date2 != -1:
                            all_years.append(date2)

                        if (date1 >= 0 and date2 >= 0) and (date2 < date1):
                            date2 = date1

                        if date1 >= 0 and date2 >= 0:
                            all_intervals.append((date1, date2))
                        elif date1 >= 0:
                            all_intervals.append((date1, YEARMAX))  # date1))
                        elif date2 >= 0:
                            all_intervals.append((YEARMIN, date2))  # ,date2))

                        time_str = '\t'.join(l[3:])

                    if time_str not in timeStr2Id:
                        newId = len(timeStr2Id)
                        timeStr2Id[time_str] = newId
                        id2TimeStr[newId] = time_str

        if "####-##-##\t####-##-##" not in timeStr2Id:
            timeStr2Id["####-##-##\t####-##-##"] = len(timeStr2Id)
            id2TimeStr[timeStr2Id["####-##-##\t####-##-##"]] = "####-##-##\t####-##-##"

        # all_years.append(self.unk_time_str)
        # all_intervals.append((self.unk_time_str,self.unk_time_str))
        all_years.append(YEARMIN)
        all_years.append(YEARMAX)

        #for index, year in enumerate(sorted(list(set(all_years)))):
        #    dateYear2id[year] = index
         #   id2dateYear[index] = year

        #dateYear2id[self.unk_time_str] = len(dateYear2id)
        #id2dateYear[dateYear2id[self.unk_time_str]] = self.unk_time_str

        all_intervals.append((YEARMIN, YEARMAX))
        for index, year_tup in enumerate(sorted(list(set(all_intervals)))):
            dateYears2intervalId[year_tup] = index
            intervalId2dateYears[index] = year_tup

        dateYears2intervalId[(self.unk_time_str, self.unk_time_str)] = len(
            dateYears2intervalId)  ##(year, yearmax) or (yearmin, year)
        intervalId2dateYears[dateYears2intervalId[(self.unk_time_str, self.unk_time_str)]] = (
            self.unk_time_str, self.unk_time_str)  # (0,0)#

        # print("dateYear2id:",dateYear2id)

        #return dateYear2id, id2dateYear, dateYears2intervalId, intervalId2dateYears, timeStr2Id, id2TimeStr
        return dateYears2intervalId, intervalId2dateYears, timeStr2Id, id2TimeStr

    @staticmethod
    def convert_dict2mat_tup(dict_in):
        dict_mat_s = numpy.zeros(len(dict_in))
        dict_mat_e = numpy.zeros(len(dict_in))
        # ipdb.set_trace()
        try:
            for key in dict_in.keys():
                if key == ('UNK-TIME', 'UNK-TIME'):
                    dict_mat_s[int(dict_in[key])] = -1
                    dict_mat_e[int(dict_in[key])] = -1
                else:
                    dict_mat_s[int(dict_in[key])] = key[0]
                    dict_mat_e[int(dict_in[key])] = key[1]
        except:
            pdb.set_trace()
        dict_mat_s = numpy.array(dict_mat_s)
        dict_mat_e = numpy.array(dict_mat_e)
        return dict_mat_s, dict_mat_e

    @staticmethod
    def convert_year2id(mat_in, map_y2i):
        # ipdb.set_trace()
        mat_out = numpy.zeros(mat_in.shape)
        for i in range(mat_in.shape[0]):
            if int(mat_in[i]) == -1:
                mat_out[i] = int(map_y2i['UNK-TIME'])
            else:
                #print("self.dateYear2id: {}".format(map_y2i['0']))
                #print("i: {} mat_in[i]: {} int(mat_in[i]): {}".format(i, mat_in[i], int(mat_in[i])))
                mat_out[i] = int(map_y2i[mat_in[i]])
        return mat_out
        
        
    @staticmethod
    def convert_dict2mat(dict_in):
        dict_mat = numpy.zeros(len(dict_in))
        # ipdb.set_trace()
        try:
            for key in dict_in.keys():
                if type(key) == tuple:
                    if key == ('UNK-TIME', 'UNK-TIME'):
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(numpy.mean(key))
                else:
                    if key == 'UNK-TIME':
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(key)
        except:
            pdb.set_trace()
        dict_mat = numpy.array(dict_mat)
        return dict_mat

    def check_date_validity(self, date):
        # if DATASET == 'ICEWS':
        if self.dataset.lower().startswith('icews'):
            year, month, day = date.split('-')
            # return int(year + month + day)
            return int(year)*375 + int(month)*31 + int(day)
        start = date.split('-')[0]
        if start.find('#') == -1 and len(start) == YEAR_STR_LEN:
            return int(start)
        else:
            return -1
