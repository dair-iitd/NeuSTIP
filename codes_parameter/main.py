import torch
import numpy
import copy
from knowledge_graph_utils import mask2list, list2mask, build_graph
from metrics import Metrics
from reasoning_model import ReasoningModel
from embedding import TimePlex
import collections
from collections import defaultdict
import gc
import os
import pickle
from time_prediction.evaluate_helper import prepare_data_iou_scores, load_pickle_aeiou, stack_tensor_list, get_thresholds, compute_scores, compute_preds
from time_prediction.interval_metrics import giou_score, aeiou_score, tac_score, smooth_iou_score, gaeiou_score

class NeuSTIP(ReasoningModel):
    def __init__(self, dataset, kb_dict,eta, lr,pe, use_duration, max_depth, clb, print=print):
        super(NeuSTIP, self).__init__()
        self.R = dataset['R']
        self.E = dataset['E']
        self.T = dataset['T']
        self.knowns = dataset['knowns'] # For time aware filtering

        print("NeuSTIP Init", self.E, self.R, self.T)
    

        embedding_dim = 200 # used in TimePlex
        if (dataset['name'] == 'YAGO11k'):
            srt_wt = 5
            ort_wt = 5
            sot_wt = 0
            nentity = 10619
            nrelation = 10
            pairs_wt = 3.0 # 0 for WIKIDATA12k
            recurrent_wt = 5.0
            embedding_path = '../data/YAGO11k'
            use_inverse = False
        else:
            # WIKIDATA12k
            srt_wt = 5
            ort_wt = 5
            sot_wt = 5
            nentity = 12545
            nrelation = 24
            pairs_wt = 0.0
            recurrent_wt = 5.0
            embedding_path ='../data/WIKIDATA12k'
            use_inverse = True

        self.TimePlex = TimePlex(embedding_path, embedding_dim, srt_wt, ort_wt, sot_wt, nentity, nrelation, pairs_wt, recurrent_wt, use_inverse, dataset['id2time'])
        
        self.eta = eta
        self.dataset = dataset
        self.set_args(lr,pe,max_depth)
        self.training = True
        self.cuda()
        self._print= print
        self.print = self.log_print
        self.debug = False
        self.recording = False
        self.rgnd_buffer = dict()
        self.rgnd_buffer_test = dict()
        self.clb = clb
    
        self.embedding_dim_normal = 32
        self.embedding_dim_allen = 32
        self.hidden_dim = 32 
        self.normal_embedding_rnn = torch.nn.Embedding(self.R,self.embedding_dim_normal).cuda() # Embedding for each normal predicate in body
        self.allen_embedding_rnn = torch.nn.Embedding(14,self.embedding_dim_allen).cuda() # Embedding for each Allen predicate, 14th is for NOREL (atemporal rules)
        self.rule_head_embedding_link = torch.nn.Embedding(1,self.hidden_dim) # Embedding for rule head, link
        self.rule_head_embedding_start = torch.nn.Embedding(1,self.hidden_dim) # Embedding for rule head, start time
        self.rule_head_embedding_end = torch.nn.Embedding(1,self.hidden_dim) # Embedding for rule head, end time
    
        self.gru = torch.nn.GRU(self.embedding_dim_normal + self.embedding_dim_allen,self.hidden_dim,batch_first=True).cuda()
        self.rgnd_buffer_link = dict()    
        self.rgnd_buffer_test_link = dict()

        self.rgnd_buffer_time = dict()   
        self.rgnd_buffer_test_time = dict()

        timedata = self.dataset['id2time']
        mintime = 10e9
        maxtime = -1
            
        for i in timedata.keys():
            if int(i) < mintime:
              mintime = i
            if int(i) > maxtime:
              maxtime = i
        self.mintime = mintime
        self.maxtime = maxtime - 1  # avoid UNK-time-str         
        self.totalintervals = len(timedata.keys()) - 1
        self.exampleid = dict()
        self.ktest_sub = kb_dict['test']
        self.kvalid_sub = kb_dict['valid']
        self.map_answer_start = [collections.defaultdict(dict) for i in range(self.R)]
        self.map_answer_end   = [collections.defaultdict(dict) for i in range(self.R)]
        self.use_duration = use_duration

    def log_print(self, *args, **kwargs):
        import datetime
        timestr = datetime.datetime.now().strftime("%H:%M:%S.%f")
        #if hasattr(self, 'em'):
        emstr = 0#self.em if self.em < self.num_em_epoch else '#'
        prefix = f"r = {self.r} EM = {emstr}"
        self._print(f"[{timestr}] {prefix} | ", end="")
        self._print(*args, **kwargs)

    def gru_step(self,normal_pred,allen_pred, type_eval='link'):

        ne = self.normal_embedding_rnn(normal_pred)
        ae = self.allen_embedding_rnn(allen_pred) 
        input_list = []
        for i in range(ne.shape[1]):
            #print(ne[:,i,:].shape)
            concat_embed = torch.cat([ae[:,i,:],ne[:,i,:]],dim=-1) # allen, normal
            input_list.append(concat_embed.reshape(ne.shape[0],1,-1))
    
        input_gru = torch.cat(input_list,dim=1)
        output, hidden = self.gru(input_gru)
        out_vecs = output[:,-1,:].reshape(ne.shape[0],self.hidden_dim)
        out_vecs_norm = out_vecs/ out_vecs.norm(dim=-1)[:,None] # shape -> batch_size x hidden_dim

        if (type_eval == 'link'):
            rule_head_norm = self.rule_head_embedding_link(torch.zeros(1,dtype=torch.long))/self.rule_head_embedding_link(torch.zeros(1,dtype=torch.long)).norm(dim=-1) # 1 x hidden_dim
        elif (type_eval == 'start'):
            rule_head_norm = self.rule_head_embedding_start(torch.zeros(1,dtype=torch.long))/self.rule_head_embedding_start(torch.zeros(1,dtype=torch.long)).norm(dim=-1)
        else:
            rule_head_norm = self.rule_head_embedding_end(torch.zeros(1,dtype=torch.long))/self.rule_head_embedding_end(torch.zeros(1,dtype=torch.long)).norm(dim=-1)

        cosine_similarity_normalised = ((out_vecs_norm @ rule_head_norm.t().cuda()) + 1)/2 # dimension -> batch size x 1 # brought in 0,1

        return cosine_similarity_normalised

    def get_duration_matrix(self):
        mean_ilength = self.dataset['mean_ilength'][self.r]
        var_ilength = self.dataset['var_ilength'][self.r]
        duration_matrix = torch.zeros(self.totalintervals, self.totalintervals)
        for i in range(self.totalintervals):
            for j in range(i,self.totalintervals):
                year_start = int(self.dataset['id2time'][i])
                year_end = int(self.dataset['id2time'][j])
                diff = year_end - year_start + 1
                x = -(diff - mean_ilength)**2
        
                x = (x / (2*var_ilength))
            
                duration_matrix[i][j] = torch.exp(x)

        return duration_matrix

    def _evaluate_time(self, valid_batch, type_eval ='valid',batch_size=None,all_metrics=False):
        print = self.print
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')
        
        self.eval()
        with torch.no_grad():
    
            if (type_eval == 'test'):
                kb_test = self.ktest_sub.facts
                raw_kb = self.ktest_sub
                batch_test = self.test_batch_time
            else:
                kb_test = self.kvalid_sub.facts
                raw_kb = self.kvalid_sub
                batch_test = self.valid_batch_time

            facts_test  = []
            test_time_scores_dict = dict()
            for onefact in kb_test:
                if onefact[1] == self.r:
                    facts_test.append(onefact.tolist())
            facts_test_numpy = numpy.array(facts_test, dtype='int64') 
            scores_t_list_test = []
            scores_t_list_test_start = []  
            scores_t_list_test_end   = []              
            t_answers_test    =[]

            scores_t_list_test_gr = []
            scores_t_list_test_start_gr = []  
            scores_t_list_test_end_gr   = []              
            t_answers_test_gr    =[]

            scores_t_list_test_ungr = []
            scores_t_list_test_start_ungr = []  
            scores_t_list_test_end_ungr   = []              
            t_answers_test_ungr    =[]


            for i in range(0, len(batch_test), batch_size):
                loss, scores_t_list_batch, t_answers_batch, scores_t_list_batch_gr, t_answers_batch_gr, scores_t_list_batch_ungr, t_answers_batch_ungr  = model(batch_test[i: i + batch_size],type_eval=type_eval,type_pred='time')
                              
                scores_t_list_test_start.extend(scores_t_list_batch[0]) #scores
                scores_t_list_test_end.extend(scores_t_list_batch[1]) #scores   

                scores_t_list_test_start_gr.extend(scores_t_list_batch_gr[0]) #scores
                scores_t_list_test_end_gr.extend(scores_t_list_batch_gr[1]) #scores 

                scores_t_list_test_start_ungr.extend(scores_t_list_batch_ungr[0]) #scores
                scores_t_list_test_end_ungr.extend(scores_t_list_batch_ungr[1]) #scores 
                    
                t_answers_test.extend(t_answers_batch)
                t_answers_test_gr.extend(t_answers_batch_gr) 
                t_answers_test_ungr.extend(t_answers_batch_ungr) 
             
                if i % print_epoch == 0 and i > 0:
                    print(f"eval #{i}/{len(batch_test)}")

                del scores_t_list_batch, t_answers_batch, scores_t_list_batch_gr, t_answers_batch_gr, scores_t_list_batch_ungr, t_answers_batch_ungr
                gc.collect()

            scores_t_list_test.append(scores_t_list_test_start)
            scores_t_list_test.append(scores_t_list_test_end)

            scores_t_list_test_gr.append(scores_t_list_test_start_gr)
            scores_t_list_test_gr.append(scores_t_list_test_end_gr)

            scores_t_list_test_ungr.append(scores_t_list_test_start_ungr)
            scores_t_list_test_ungr.append(scores_t_list_test_end_ungr)

            if len(t_answers_test) > 0:
                t_answer_final_test = torch.from_numpy(numpy.stack(t_answers_test, axis=0)).unsqueeze(1)
            else:
                t_answer_final_test = torch.tensor([])

            if len(t_answers_test_gr) > 0:
                t_answer_final_test_gr = torch.from_numpy(numpy.stack(t_answers_test_gr, axis=0)).unsqueeze(1)
            else:
                t_answer_final_test_gr = torch.tensor([])

            if len(t_answers_test_ungr) > 0:
                t_answer_final_test_ungr = torch.from_numpy(numpy.stack(t_answers_test_ungr, axis=0)).unsqueeze(1)
            else:
                t_answer_final_test_ungr = torch.tensor([])

       
            test_time_scores_dict = []
            test_time_scores_dict_gr =[]
            test_time_scores_dict_ungr = []                    
            if t_answer_final_test.shape[-1] > 0:
                test_time_scores_dict = prepare_data_iou_scores(t_answer_final_test, raw_kb, scores_t=scores_t_list_test, load_to_gpu=True)
                test_time_scores_dict["facts"] = facts_test_numpy                
                test_time_scores_dict["data_folder_full_path"] = self.ktest_sub.datamap.dataset_root

            if t_answer_final_test_gr.shape[-1] > 0:
                test_time_scores_dict_gr = prepare_data_iou_scores(t_answer_final_test_gr,  raw_kb, scores_t=scores_t_list_test_gr, load_to_gpu=True)
                test_time_scores_dict_gr["facts"] = facts_test_numpy                
                test_time_scores_dict_gr["data_folder_full_path"] = self.ktest_sub.datamap.dataset_root

            if t_answer_final_test_ungr.shape[-1] > 0:
                test_time_scores_dict_ungr = prepare_data_iou_scores(t_answer_final_test_ungr,  raw_kb, scores_t=scores_t_list_test_ungr, load_to_gpu=True)
                test_time_scores_dict_ungr["facts"] = facts_test_numpy                
                test_time_scores_dict_ungr["data_folder_full_path"] = self.ktest_sub.datamap.dataset_root

            if len(test_time_scores_dict) > 0:
                path = "./workspace/aeIOU/r"+str(self.r)+"/"
             
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
               
                scores_dict, aeIOUat1, numofexamples = self.compute_interval_scores(test_time_scores_dict, method='start-end-exhaustive-sweep', save_time_results=path, type_eval=type_eval, all_metrics=all_metrics,gr_type='all')
                del test_time_scores_dict
            else:
                aeIOUat1 = torch.zeros(1)

            if len(test_time_scores_dict_gr) > 0:
                path = "./workspace/aeIOU/r"+str(self.r)+"/"
          
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
               
                scores_dict_gr, aeIOUat1_gr, numofexamples = self.compute_interval_scores(test_time_scores_dict_gr, method='start-end-exhaustive-sweep', save_time_results=path, type_eval=type_eval, all_metrics=all_metrics,gr_type='gr')
                del test_time_scores_dict_gr
            else:
                aeIOUat1_gr = torch.zeros(1)

            if len(test_time_scores_dict_ungr) > 0:
                path = "./workspace/aeIOU/r"+str(self.r)+"/"
                #print(path)
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
        
                scores_dict_ungr, aeIOUat1_ungr, numofexamples = self.compute_interval_scores(test_time_scores_dict_ungr, method='start-end-exhaustive-sweep', save_time_results=path, type_eval=type_eval, all_metrics=all_metrics,gr_type='ungr')
                del test_time_scores_dict_ungr
            else:
                aeIOUat1_ungr = torch.zeros(1)
        
    
        gc.collect()
     
        return aeIOUat1, aeIOUat1_gr, aeIOUat1_ungr

    def compute_interval_scores(self,test_time_scores_dict, save_time_results=None, method='greedy-coalescing', type_eval='valid', all_metrics=False,gr_type='all'):
        """
        Takes input time scores stored in test_pickle and valid_pickle (for test and valid KBs respectively)
        Using these time scores, depending on method it predicts intervals for each fact in test kb (a ranking ideally instead of a single interval),
        and returns gIOU/aeIOU/IOU scores @1/5/10
        """
        print = self.print
        thresholds = None
        aeIOUat1 = -1
        # load from dict
        t_scores, duration_scores, facts, dataset_root, use_time_interval, id_year_map, t_gold_min, t_gold_max = load_pickle_aeiou(test_time_scores_dict)
      
        print("Using method {}".format(method))

       
        if method in ['start-end-exhaustive-sweep']:  # for time boundary models
            #print("* t_scores: {}".format(t_scores))
            start_t_scores, end_t_scores = t_scores

            start_t_scores = stack_tensor_list(start_t_scores)
            end_t_scores = stack_tensor_list(end_t_scores)

            t_scores = (start_t_scores, end_t_scores)


        id_year_map = id_year_map.long() #test_kb.datamap.binId2year_mat = self.convert_dict2mat(self.year2id)

        id_year_map_dict = {}
        for i, j in enumerate(id_year_map):
            id_year_map_dict[i] = j             
       
        print("**************")
        topk_ranks = 1

        # score_func = {"precision":precision_score, "recall":recall_score, "aeIOU": aeiou_score, "TAC": tac_score, "IOU": smooth_iou_score, "gIOU": giou_score}
        if (all_metrics):
            score_func = {"gIOU": giou_score, "aeIOU": aeiou_score, "TAC": tac_score, "IOU": smooth_iou_score, "gaeIOU": gaeiou_score}
        else:
            score_func = {"aeIOU":aeiou_score}

        scores_dict = {}  # for saving later

        for score_to_compute in score_func.keys():
    
            
            iou_scores = compute_scores(f'./workspace/aeIOU/r{self.r}/'+self.threshold_filename,id_year_map, facts, t_gold_min, t_gold_max, t_scores,  self.no_grnd_start_year, self.no_grnd_end_year,self.mintime + 1, self.maxtime-1, self.mean_offset, self.duration_matrix, self.use_duration,method=method,
                                        thresholds=thresholds,
                                        score_func=score_func[score_to_compute], topk_ranks=topk_ranks,type_eval = type_eval)

            # output best iou @ k
            iouatk = [1]


            for i in iouatk:
                all_scores = torch.stack(iou_scores[:i]) #all_scores = torch.Size([i = 1, test_data_size]), when i =1, all_scores = torch.Size([i = 5, test_data_size]), when i =5, all_scores = torch.Size([i = 10, test_data_size]), when i =10
                best_scores, _ = torch.max(all_scores, 0) # best_scores.shape: torch.Size([test_data_size])
                scores_dict[(i, score_to_compute)] = best_scores

                #     print("best_scores shape:",best_scores.shape)
                if (type_eval == 'test'):
                    print("Best {} @{}: {}".format(score_to_compute, i, torch.mean(best_scores)))

      
               
                if i == 1 and score_to_compute == "aeIOU":
                    aeIOUat1 = torch.mean(best_scores)
                
        if (save_time_results is not None):
            # saves metrics of the form
            pickle_filename = f'./workspace/aeIOU/r{self.r}/'+self.threshold_filename+'_'+gr_type+'_'+type_eval+'_time_score_analysis'
            threshold_filename = "{}_valid_threshold".format(save_time_results)


            with open(pickle_filename, 'wb') as handle:
                #pickle.dump({'facts': facts, 'scores_dict': scores_dict, 'time_scores':t_scores, 't_gold_min': gold_min, 't_gold_max': gold_max,
                #             't_pred_min': pred_min, 't_pred_max': pred_max}, handle)
                pickle.dump({'scores_dict': scores_dict,'aeIOU@1':aeIOUat1, 'numberoffacts':start_t_scores.shape[0]}, handle)
      
            
        else:
            print("\nNot saving scores")
            
        return scores_dict, aeIOUat1, start_t_scores.shape[0]

    def forward(self, batch, type_eval='valid', type_pred = 'link'): 
        #print(self.metrics.get_N())
        print = self.print
        E = self.E
        R = self.R


        if (type_pred!='time'): # Link is there
           
            rule_weight_link = []
            for i in range(self.MAX_RULE_LEN): # Batching based on rule length
                if (self.rules_length_wise[i].shape[0] > 0):
                    rule_weight_i_link = self.gru_step(self.rules_length_wise[i]%(self.R), self.rules_length_wise[i]//(self.R), 'link')
                    rule_weight_link.append(rule_weight_i_link)

           
            rule_weight_link = (torch.cat(rule_weight_link,dim=0).squeeze(1))*self.prior_link

            crule_link = []
            crule_weight_link = []
            centity = []
            ccount_link = []
            csplit_link = [0]

            for single in batch:
                _, _, _, _,_,_,_crule_link, _, _, _centity, _, _, _ccount_link, _, _ = self.load_batch(single) # ccount is the number of groundings
                #print(_crule_link, _centity)
                if _crule_link.size(0) == 0:
                    csplit_link.append(csplit_link[-1])
                    continue

                crule_link.append(_crule_link)
                crule_weight_link.append(rule_weight_link.index_select(0, _crule_link))
                centity.append(_centity)
                ccount_link.append(_ccount_link)
                csplit_link.append(csplit_link[-1] + _crule_link.size(-1))
              
            if len(centity) == 0:
                crule_link = torch.tensor([]).long().cuda()
                crule_weight_link = torch.tensor([]).cuda()
                centity = torch.tensor([]).long().cuda()
                ccount_link = torch.tensor([]).long().cuda()
                cscore = torch.tensor([]).cuda()
            
            else:
                crule_link = torch.cat(crule_link, dim=0)
                crule_weight_link = torch.cat(crule_weight_link, dim=0)
                centity = torch.cat(centity, dim=0)
                ccount_link = torch.cat(ccount_link, dim=0)
                cscore = ccount_link * (crule_weight_link.clamp(min=1e-5))

            result_link = []
            result_link_grounded = []
            result_link_ungrounded = []
            for i, single in enumerate(batch):
                element, link_answer, link_t_list, time_answer, time_t_list,masked_link, _crule_link, crule_start, crule_end, _centity, cstart, cend, _ccount_link, ccount_start, ccount_end = self.load_batch(single)

                _h = (element[0],element[2],element[3])
                t_list = link_t_list

                mask = masked_link
                grounded = True

                if _crule_link.size(0) != 0:
                    crange = torch.arange(csplit_link[i], csplit_link[i + 1]).cuda()
                    score = torch.sparse.sum(torch.sparse.FloatTensor(
                        torch.stack([_centity, _crule_link], dim=0),
                        self.index_select(cscore, crange),
                        torch.Size([E, self.num_rule])
                    ), -1).to_dense()
                else:
                    grounded = False
                    score = torch.zeros(self.E).cuda()

                    score.requires_grad_()


                if self.recording:
                    self.record.append((score.cpu(), mask, t_list))

                elif not self.training:
               
                    for t in t_list:
                        kge_score = torch.zeros(self.E).cuda()
                        total_score = score
            
                        if (self.eta > 0):
                            kge_score = (self.TimePlex(torch.tensor(_h[0]).unsqueeze(0).cuda(),torch.tensor(self.r).unsqueeze(0).cuda(),None,torch.tensor(_h[1]).unsqueeze(0).cuda(),torch.tensor(_h[2]).unsqueeze(0).cuda(),False,False)).squeeze(0)
                        
                            total_score+= self.eta_learnable.cuda()*(kge_score[:-1])


                        
                        # TIME AWARE FILTERING DONE BELOW (similar to TimePlex, verified)
                        h_repeated = torch.tensor([_h[0]]*(_h[2]-_h[1]+1)).cuda()
                        r_repeated = torch.tensor([self.r]*(_h[2]-_h[1]+1)).cuda()
                        t_repeated = torch.tensor([t]*(_h[2]-_h[1]+1)).cuda()
                        t1_repeated = torch.tensor([_h[1]]*(_h[2]-_h[1]+1)).cuda()
                        total_score_expected = total_score.unsqueeze(0).gather(1,torch.tensor([t]).unsqueeze(0).cuda())
        
                        t_start = _h[1]
                        t_end = _h[2]
                        for t_point in range(t_start,t_end+1):
                            t1_repeated[t_point-t_start] = t_point

                        total_score_repeated = (total_score.unsqueeze(0)).repeat((_h[2]-_h[1]+1),1) # IL X ES expected dimension
    

                        total_score_expected_repeated = (total_score_expected).repeat((_h[2]-_h[1]+1),1) # IL x 1 expected dimension
            
                        knowns_filter = torch.from_numpy(self.get_knowns(h_repeated,r_repeated,t1_repeated,self.knowns)).cuda()
                        # self.knowns obtained similar to TimePlex, verified
                        minimum_value = -(200*200) 
                        total_score_repeated.scatter_(1,knowns_filter,minimum_value)
                        
                        greater = (total_score_repeated > total_score_expected_repeated) # IL x ES

                        greater_eq = (total_score_repeated >= total_score_expected_repeated) # IL x ES ITSELF WILL ALSO BE FILTERED OUT IN KNOWNS
                    
                        expected_ranks = (greater.sum(dim=1) + greater_eq.sum(dim=1))/2 + 1
                       
                        rank_avg = expected_ranks.sum(dim=0)/float((_h[2]-_h[1]+1))
                        
                        mr = rank_avg
                        mrr = 1/rank_avg
                        h1 = (rank_avg <=1)
                        h3 = (rank_avg <=3)
                        h10 = (rank_avg<=10)

        
                        result_link.append((1,mr,mrr,h1,h3,h10))
                        # if (grounded): # Commented for efficiency
                        #     result_link_grounded.append((1,mr,mrr,h1,h3,h10))
                        # else:
                        #     result_link_ungrounded.append((1,mr,mrr,h1,h3,h10))
                   

                if score.dim() == 0:
                    continue
                
                kge_score= torch.zeros(self.E).cuda()

                if (self.eta > 0):
        
                    with torch.no_grad():
                        kge_score = (self.TimePlex(torch.tensor(_h[0]).unsqueeze(0).cuda(),torch.tensor(self.r).unsqueeze(0).cuda(),None,torch.tensor(_h[1]).unsqueeze(0).cuda(),torch.tensor(_h[2]).unsqueeze(0).cuda(),False,False)).squeeze(0)
                        kge_score = kge_score[:-1]
                   
                score = self.eta_learnable.cuda()*(kge_score) + score
                
                score = score.softmax(dim=-1)
                

                loss_link = torch.tensor(0.0).cuda().requires_grad_() + 0.0
                neg = score.masked_select(~mask.bool())
                eps = 1e-9

                loss_link += neg.sum()
            

                for t in t_list:
                    s = score[t]
                    wrong = (neg > s)
                    loss_link += ((neg - s) * wrong).sum() / wrong.sum().clamp(min=1)

   
        if (type_pred!='link'): 

            def Extract(lst, i):
                return [item[i] for item in lst]

            rule_weight_start = []
            rule_weight_end = []

            for i in range(self.MAX_RULE_LEN):
                if (self.rules_length_wise[i].shape[0] > 0):
                    rule_weight_i_start = self.gru_step(self.rules_length_wise[i]%(self.R), self.rules_length_wise[i]//(self.R), 'start')
                    rule_weight_i_end = self.gru_step(self.rules_length_wise[i]%(self.R), self.rules_length_wise[i]//(self.R), 'end')
                    rule_weight_start.append(rule_weight_i_start)
                    rule_weight_end.append(rule_weight_i_end)

       
          
            rule_weight_start = (torch.cat(rule_weight_start,dim=0).squeeze(1))*self.prior_start
            rule_weight_end = (torch.cat(rule_weight_end,dim=0).squeeze(1))*self.prior_end

            crule_start = []
            crule_end = []

            crule_weight_start = []
            crule_weight_end = []
        
            cstart = []
            ccount_start = []
            cend = []
            ccount_end = []

            csplit_start = [0]
            csplit_end = [0]

            data_pickle = dict()
     
            for single in batch:
                _, _,_, _,_,_, _, _crule_start, _crule_end, _, _cstart, _cend, _, _ccount_start, _ccount_end = self.load_batch(single)

                if _crule_start.size(0) == 0:
                    csplit_start.append(csplit_start[-1])
                else:
                    crule_start.append(_crule_start)                                     # rules
                    crule_weight_start.append(rule_weight_start.index_select(0, _crule_start)) # rule weights
                    cstart.append(_cstart)                                               # time instances
                    ccount_start.append(_ccount_start)                                   # distribution
                    csplit_start.append(csplit_start[-1] + _crule_start.size(-1))
                
                if _crule_end.size(0) == 0:
                    csplit_end.append(csplit_end[-1])
                else:
                    crule_end.append(_crule_end)                                         # rules
                    crule_weight_end.append(rule_weight_end.index_select(0, _crule_end))     # rule weights
                    cend.append(_cend)                                                   # time instances
                    ccount_end.append(_ccount_end)                                       # distribution
                    csplit_end.append(csplit_end[-1] + _crule_end.size(-1))


            if len(cstart) == 0:
                crule_start = torch.tensor([]).long().cuda()      # rules
                crule_weight_start = torch.tensor([]).cuda()      # rule weights
                cstart = torch.tensor([]).long().cuda()           # time instances

                ccount_start = torch.tensor([]).long().cuda()     # distribution
                cscore_start = torch.tensor([]).cuda()
            else:
                crule_start = torch.cat(crule_start, dim=0)
                crule_weight_start = torch.cat(crule_weight_start, dim=0)
                cstart = torch.cat(cstart, dim=0)
                ccount_start = torch.cat(ccount_start, dim=0)           
                cscore_start = ccount_start * crule_weight_start.clamp(min=0.0)  

            if len(cend) == 0: # REPLCACE ALL BY END
                crule_end = torch.tensor([]).long().cuda() # rules
                crule_weight_end = torch.tensor([]).cuda() # rule weights
                cend = torch.tensor([]).long().cuda()      # time instances
                
                ccount_end = torch.tensor([]).long().cuda() # distribution
                cscore_end = torch.tensor([]).cuda()
            else:
                crule_end = torch.cat(crule_end, dim=0)      # rules
                crule_weight_end = torch.cat(crule_weight_end, dim=0) # rule weights
                cend = torch.cat(cend, dim=0)
                ccount_end = torch.cat(ccount_end, dim=0)
                cscore_end = ccount_end * crule_weight_end.clamp(min=0.0)          

            result_time = []
            
            scores_t_list = []
            scores_t_list_start = []
            scores_t_list_end   = []
            t_answers    =[]

            scores_t_list_gr = []
            scores_t_list_start_gr = []
            scores_t_list_end_gr   = []
            t_answers_gr    =[]

            scores_t_list_ungr = []
            scores_t_list_start_ungr = []
            scores_t_list_end_ungr   = []
            t_answers_ungr    =[]

            for i, single in enumerate(batch):

                element, link_answer, link_t_list, time_answer, time_t_list, masked_link, crule_link, _crule_start, _crule_end, centity, _cstart, _cend, ccount_link, ccount_start, ccount_end = self.load_batch(single)
                _h = (element[0], element[1])
           
                start_not_grounded = True
                end_not_grounded = True
                if _crule_start.size(0) != 0:
                    crange_start = torch.arange(csplit_start[i], csplit_start[i + 1]).cuda()

                    score_start = torch.sparse.sum(torch.sparse.FloatTensor(
                       torch.stack([_cstart, _crule_start], dim=0),
                       self.index_select(cscore_start, crange_start),
                       torch.Size([self.totalintervals, self.num_rule]) # cscore is of this size (319 scores)
                    ), -1).to_dense()

                else:
                    
                    score_start = torch.zeros(self.totalintervals).cuda()

                    score_start.requires_grad_()  

                if (score_start[time_answer[0]]!=0):
                    start_not_grounded = False



                if _crule_end.size(0) != 0:
                    crange_end = torch.arange(csplit_end[i], csplit_end[i + 1]).cuda()
    
                    score_end = torch.sparse.sum(torch.sparse.FloatTensor(
                        torch.stack([_cend, _crule_end], dim=0),
                        self.index_select(cscore_end, crange_end),
                         torch.Size([self.totalintervals, self.num_rule]) # cscore is of this size (319 scores)
                    ), -1).to_dense()

                else:

                    score_end = torch.zeros(self.totalintervals).cuda()
                
                    score_end.requires_grad_()  

                if (score_end[time_answer[1]]!=0):
                    end_not_grounded = False
              

                if self.recording:
                    self.record.append((score.cpu(), mask, t_list))
                

                elif not self.training:

                    kge_score = torch.zeros(self.totalintervals).cuda()
                    if (self.eta > 0):
                    
                        kge_score = (self.TimePlex(torch.tensor(_h[0]).unsqueeze(0).cuda(),torch.tensor(self.r).unsqueeze(0).cuda(), torch.tensor(_h[1]).unsqueeze(0).cuda(),  torch.tensor(time_answer[0]).unsqueeze(0).cuda(),torch.tensor(time_answer[2]).unsqueeze(0).cuda(),True,False)).squeeze(0)
                        kge_score = kge_score[:-3] # Extra id, UNK time str, yearmax not used

                    total_score_start = self.eta_learnable.cuda()*(kge_score)+ score_start 
                    total_score_end = self.eta_learnable.cuda()*(kge_score)+ score_end

                    # if (start_not_grounded and end_not_grounded): # Commented for efficiency
                    #     scores_t_list_start_ungr.append(total_score_start)
                    #     scores_t_list_end_ungr.append(total_score_end)
                    #     t_answers_ungr.append(numpy.array(time_answer))
                    # else:
                    #     scores_t_list_start_gr.append(total_score_start)
                    #     scores_t_list_end_gr.append(total_score_end)
                    #     t_answers_gr.append(numpy.array(time_answer))

                    scores_t_list_start.append(total_score_start)
                    scores_t_list_end.append(total_score_end)

                    del total_score_start, total_score_end

                    

                    t_answers.append(numpy.array(time_answer))
                    

                if (self.eta > 0):
                    with torch.no_grad():
                        kge_score = (self.TimePlex(torch.tensor(_h[0]).unsqueeze(0).cuda(),torch.tensor(self.r).unsqueeze(0).cuda(), torch.tensor(_h[1]).unsqueeze(0).cuda(),  torch.tensor(time_answer[0]).unsqueeze(0).cuda(),torch.tensor(time_answer[2]).unsqueeze(0).cuda(),True,False)).squeeze(0)
                        kge_score = kge_score[:-3] # Extra id, UNK time str, yearmax not used

                    score_start = score_start+ self.eta_learnable.cuda()*(kge_score)
                    score_end = score_end + self.eta_learnable.cuda()*(kge_score)



                score_start = score_start.softmax(dim=-1)
                score_end   = score_end.softmax(dim=-1)
                  
                loss_time = torch.tensor(0.0).cuda().requires_grad_() + 0.0
                t_list = time_t_list # we have handled it for train as well here

                mask_start = list2mask(Extract(t_list, 0),  self.totalintervals).to(loss_time.get_device())
                # mask_start -> These are the true start times
                neg_start = score_start.masked_select(~mask_start.bool())
                mask_end = list2mask(Extract(t_list, 1),  self.totalintervals).to(loss_time.get_device())
                neg_end  = score_end.masked_select(~mask_end.bool())
                
                eps = 1e-9




                
                t_list_start = Extract(t_list, 0)
                for t_list_start_i in t_list_start:
                    weight = ((torch.tensor(t_list_start_i) - torch.range(0,self.totalintervals-1))/(self.totalintervals-1)).cuda()
                   
                    loss_time += (-1*((score_start[t_list_start_i] - neg_start)*(weight.masked_select(~mask_start.bool()))).sum())/((weight.masked_select(~mask_start.bool())).sum().clamp(min=1))            
                
           

                t_list_end = Extract(t_list, 1)  
                for t_list_end_i in t_list_end:
                    weight = ((torch.tensor(t_list_end_i) - torch.range(0,self.totalintervals-1))/(self.totalintervals-1)).cuda()
                    
                    loss_time += (-1*((score_end[t_list_end_i] - neg_end)*(weight.masked_select(~mask_end.bool()))).sum())/((weight.masked_select(~mask_end.bool())).sum().clamp(min=1))
                    

            scores_t_list.append(scores_t_list_start)
            scores_t_list.append(scores_t_list_end)

            scores_t_list_gr.append(scores_t_list_start_gr)
            scores_t_list_gr.append(scores_t_list_end_gr)

            scores_t_list_ungr.append(scores_t_list_start_ungr)
            scores_t_list_ungr.append(scores_t_list_end_ungr)


            del scores_t_list_start, scores_t_list_end , scores_t_list_start_gr, scores_t_list_end_gr, scores_t_list_start_ungr, scores_t_list_end_ungr, score_start, score_end       
            

        if (type_pred == 'link'):
          
            return loss_link / len(batch), self.metrics.summary(result_link), self.metrics.summary(result_link_grounded), self.metrics.summary(result_link_ungrounded)
            
        else:
            return loss_time / len(batch), scores_t_list, t_answers, scores_t_list_gr, t_answers_gr, scores_t_list_ungr, t_answers_ungr


    def _evaluate_link(self, valid_batch, type_eval ='valid',batch_size=None):
        model = self
        if batch_size is None:
            batch_size = self.arg('predictor_batch_size') * self.arg('predictor_eval_rate')
        print_epoch = self.arg('predictor_print_epoch') * self.arg('predictor_eval_rate')
        
        self.eval()
        with torch.no_grad():
            result = Metrics.zero_value()
            result_gr = Metrics.zero_value()
            result_ungr = Metrics.zero_value()
            #print(valid_batch)
            for i in range(0, len(valid_batch), batch_size):
                # Make model do different things based on the passed Flag
                loss, cur, cur_gr, cur_ungr = model(valid_batch[i: i + batch_size],type_eval,type_pred='link')
                
                result = Metrics.merge(result, cur)
        
                if (cur_gr[0] > 0):
                    result_gr = Metrics.merge(result_gr, cur_gr)
                if (cur_ungr[0] > 0):
                    result_ungr = Metrics.merge(result_ungr, cur_ungr)
                if i % print_epoch == 0 and i > 0:
                    print(f"eval #{i}/{len(valid_batch)}")
        return result, result_gr, result_ungr

    def train_model(self):
        # self.make_batchs()
        train_batch = self.train_batch
        valid_batch = self.valid_batch
        test_batch = self.test_batch
        valid_batch_time = self.valid_batch_time
        test_batch_time = self.test_batch_time

        model = self
        print = self.print
        #print(self.metrics.get_N())
        batch_size = self.arg('predictor_batch_size')
        num_epoch = self.arg('predictor_num_epoch')  # / batch_size
        lr = self.arg('predictor_lr')  # * batch_size
        print_epoch = self.arg('predictor_print_epoch')
        valid_epoch = self.arg('predictor_valid_epoch')

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr / 5)

        if (self.train_mode == 'link'):
            self.best= Metrics.init_value() 
        else:
            self.best = 0

        self.best_model = self.state_dict()

        def train_step(batch):
            self.train()
            #print(self.train_mode)
            if (self.train_mode == 'link'):
                loss, _, _, _= self(batch,type_pred=self.train_mode) # train mode can be link, time or both
            elif (self.train_mode == 'both'):
                loss = self(batch,type_pred=self.train_mode)
            else:
                loss, _, _, _, _, _, _ = self(batch,type_pred=self.train_mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            return loss

        def metrics_score(result):
            result = Metrics.pretty(result)
            mr = result['mr']
            mrr = result['mrr']
            h1 = result['h1']
            h3 = result['h3']
            h10 = result['h10']

            def eval_ctx(local_ctx):
                local_ctx['self'] = self
                return lambda x: eval(x, globals(), local_ctx)

            return self.arg('metrics_score_def', apply=eval_ctx(locals()))

        def format(result):
            s = ""
            for k, v in Metrics.pretty(result).items():
                if k == 'num':
                    continue
                s += k + ":"
                s += "%.4lf " % v
            return s

        def valid(all_metrics):
            updated = False
            if (self.train_mode == 'link'):
                result, result_gr, result_ungr = self._evaluate_link(valid_batch,type_eval='valid')
                if metrics_score(result) > metrics_score(self.best):
                    updated = True
                    self.best = result
                    self.best_model = copy.deepcopy(self.state_dict())
             
                return updated, result, result_gr, result_ungr

            if (self.train_mode == 'time'):
                result, result_gr, result_ungr = self._evaluate_time(valid_batch_time,type_eval='valid',all_metrics=all_metrics)
                if (result.item() >= self.best):

                    updated = True
                    self.best = result.item()
                    self.best_model = copy.deepcopy(self.state_dict())
               

                return updated, result.item(), result_gr.item(), result_ungr.item()

        last_update = 0
        cum_loss = 0

        valid(False)


        if len(train_batch) == 0:
            num_epoch = 0
        for epoch in range(1, num_epoch + 1):
            if epoch % max(1, len(train_batch) // batch_size) == 0:
                from random import shuffle
                shuffle(train_batch)
            batch = [train_batch[(epoch * batch_size + i) % len(train_batch)] for i in range(batch_size)]
            loss = train_step(batch)
            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_predictor #{epoch} lr = {lr_str}")
                cum_loss *= 0


            if epoch % valid_epoch == 0:
                if valid(False)[0]:
                    last_update = epoch


        with torch.no_grad():
            self.load_state_dict(self.best_model)
        

            if (self.train_mode=='link'):
                updated, result, result_gr, result_ungr = valid(True) 
                best = result
                best_gr = result_gr
                best_ungr = result_ungr
                test_link, test_link_gr, test_link_ungr = self._evaluate_link(test_batch,type_eval='test') 
                metrics = ['mr','mrr','h@1','h@3','h@10']
               
                if (not os.path.exists(f'./workspace/metrics/r{self.r}/')):
                    os.makedirs(f'./workspace/metrics/r{self.r}/')



                valid_dict = dict()
                test_dict = dict()

                valid_dict['valid_data_size'] = int(best[0])
                valid_dict['valid_grounded_data_size'] = int(best_gr[0])
                valid_dict['valid_ungrounded_data_size'] = int(best_ungr[0])

                test_dict['test_data_size'] = int(test_link[0])
                test_dict['test_grounded_data_size'] = int(test_link_gr[0])
                test_dict['test_ungrounded_data_size'] = int(test_link_ungr[0])

                for i in range(len(metrics)):
                    valid_dict[metrics[i]] = float(best[i+1])
                    test_dict[metrics[i]] = float(test_link[i+1])
                    test_dict['grounded_'+metrics[i]] = float(test_link_gr[i+1])
                    test_dict['ungrounded_'+metrics[i]] = float(test_link_ungr[i+1])
                    valid_dict['grounded_'+metrics[i]] = float(best_gr[i+1])
                    valid_dict['ungrounded_'+metrics[i]] = float(best_ungr[i+1])
                    

                with open(f'./workspace/metrics/r{self.r}/valid_'+self.metrics_filename, 'wb') as f:
                    pickle.dump(valid_dict, f)

                with open(f'./workspace/metrics/r{self.r}/test_'+self.metrics_filename, 'wb') as f:
                    pickle.dump(test_dict, f)

              
             
                return best, test_link

            if (self.train_mode=='time'):
                updated, result, result_gr, result_ungr = valid(True) 
                best = result
                test_time, test_time_gr, test_time_ungr = self._evaluate_time(test_batch_time,type_eval='test',all_metrics=True) 
                return best, test_time


    def train_model_start(self, r, rule_file=None, model_file=None, threshold_file=None, train_mode = 'link'):
        if rule_file is None:
            rule_file = f"rules_{r}.txt"
        if model_file is None:
            model_file = f"model_{r}.pth"
        self.r = r
        self.train_mode = train_mode
        print = self.log_print
     
        rgnd_buffer_link = dict()    #Handle these buffers
        rgnd_buffer_test_link = dict()

        rgnd_buffer_time = dict()    #Handle these buffers
        rgnd_buffer_test_time = dict()

        self.duration_matrix = self.get_duration_matrix()
        self.relation_init(r=r, rule_file=rule_file) 
        self.rgnd_buffer_link = rgnd_buffer_link   # Handle these buffers
        self.rgnd_buffer_test_link = rgnd_buffer_test_link

        self.rgnd_buffer_time = rgnd_buffer_time   # Handle these buffers
        self.rgnd_buffer_test_time = rgnd_buffer_test_time

        self.metrics_filename = threshold_file # See this also
        self.threshold_filename = threshold_file

        self.no_grnd_start_year = min(max(int(self.dataset['no_grnd_start_year'][self.r]),self.mintime+1),self.maxtime-1)
        self.no_grnd_end_year = min(max(int(self.dataset['no_grnd_end_year'][self.r]),self.mintime+1),self.maxtime-1)
        self.mean_offset = int(self.dataset['no_grnd_end_year'][self.r] - self.dataset['no_grnd_start_year'][self.r])

        valid, test = self.train_model()
  

        ckpt = {
            'r': r,
            'metrics': {
                'valid': valid,
                'test': test
            },
            'predictor': self.state_dict(),
        }
        
        gc.collect()
        return valid, test


    def train(self, mode=True):
        self.training = mode
        super(NeuSTIP, self).train(mode)

    def eval(self):
        self.train(False)

    def index_select(self, tensor, index):
        if self.training:
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index)
            index = index.to(tensor.device)
            return tensor.index_select(0, index).squeeze(0)
        else:
            return tensor[index]

    @staticmethod
    def load_batch(batch):
        return tuple(map(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x, batch))


    # Init rules
    def set_rules(self, rules):
        paths = rules
        r = self.r
        self.eval()

        self.MAX_RULE_LEN = self.arg('max_rule_len') # WILL SET TO 3 HERE
        pad = (self.R - 1)*(13)
        gen_end = pad
        gen_pad = pad + 1 
        rules = []
        rules_exp = []
        # sort paths by their lengths, decreasing/increasing
        self.rules_length_wise = [] # list of tensors
        length_wise_rules = [[] for i in range(self.MAX_RULE_LEN)]

        # ALL RULES MUST BE IN INCREASING RULE LENGTH ORDER FOR CONSISTENCY DURING INDEXING OPERATION
       
        for path in paths:
            length_wise_rules[len(path)-1].append(path) # NO PADDING HERE

        for i in range(self.MAX_RULE_LEN):
            self.rules_length_wise.append(torch.LongTensor(length_wise_rules[i]).cuda())
            for j in range(len(length_wise_rules[i])):
                rules_exp.append(tuple(length_wise_rules[i][j]))
                npad = (self.MAX_RULE_LEN - (i+1))
                rules.append(length_wise_rules[i][j] + (pad,) * npad)
        # Maintain a list of self.rules
        #print(self.rules_length_wise)
        self.rules = torch.LongTensor(rules).t().cuda()
        self.rules_exp = tuple(rules_exp)


    @property
    def num_rule(self):
        if (self.rules.shape[0] == 0):
            return 0
        return self.rules.size(1)

    def allen_relation(self,t1, t2, cur_t1, cur_t2):
        
        if (t2 < cur_t1): # before
            return 0
        if (cur_t2 < t1):
            return 1
        if (self.clb == 1): # Everything else is classified as 4 (Overlaps) in case of clubbed predicates
            return 4 
        if (t1 == cur_t1 and t2 == cur_t2):
            return 12
        if (t2 == cur_t1):
            return 2
        if (cur_t2 == t1):
            return 3
        if (t2 > cur_t1 and t2 < cur_t2 and t1 < cur_t1):
            return 4
        if (cur_t2 > t1 and cur_t2 < t2 and cur_t1 < t1):
            return 5
        if (t1 == cur_t1 and t2 < cur_t2):
            return 6
        if (cur_t1 == t1 and cur_t2 < t2):
            return 7
        if (t1 > cur_t1 and t2 < cur_t2):
            return 8
        if (cur_t1 > t1 and cur_t2 < t2):
            return 9
        if (t1 > cur_t1 and t2 == cur_t2):
            return 10
        if (cur_t1 > t1 and t2 == cur_t2):
            return 11

    def get_start_end_ranges(self, allen_pred, e1, e2):
        if (allen_pred == 0): # BEFORE
            return self.mintime, e1, self.mintime, e1
        elif (allen_pred == 1): #!BEFORE
            return e2 + 1, self.maxtime +1, e2 +1, self.maxtime + 1
        elif (allen_pred == 2): #MEETS
            return self.mintime, e1+1, e1, e1 +1
        elif (allen_pred == 3): # !meets
            return e2, e2 +1 , e2, self.maxtime+1

        elif (allen_pred == 4): # overlaps
            if (self.clb == 1):
                # In case of clubbed predicates, union of all other
                return self.mintime, e2 +1, e1, self.maxtime + 1
            else:
                return self.mintime, e1, e1+1, e2

        elif (allen_pred == 5):
            return  e1+1, e2, e2+1, self.maxtime+1

        elif (allen_pred == 6):
            return e1, e1 +1, e1, e2

        elif (allen_pred == 7):
            return e1, e1+1, e2+1, self.maxtime

        elif (allen_pred == 8):
            return e1+1, e2,  e1+1, e2

        elif (allen_pred == 9):
            return self.mintime, e1,  e2+1, self.maxtime+1

        elif (allen_pred == 10):
            return e1+1, e2+1, e2, e2 + 1

        elif (allen_pred == 11):
            return self.mintime, e1, e2, e2 +1

        elif (allen_pred == 12):
            return e1, e1 +1, e2, e2 +1

    def potential_distribution_for_end_points(self, allen_pred, normal_pred, count, e1, e2):
    
        map_answer_start_local = dict()
        map_answer_end_local   = dict()
        
        start_mean_diff = self.dataset['mean_r_r_start'][self.r][normal_pred]
      
        start_var_diff   = self.dataset['var_r_r_start'][self.r][normal_pred]
        
        end_mean_diff = self.dataset['mean_r_r_end'][self.r][normal_pred]

        end_var_diff   =self.dataset['var_r_r_end'][self.r][normal_pred]

        start_rec_mean = self.dataset['mean_r_start'][self.r]
        start_rec_var = self.dataset['var_r_start'][self.r]

        end_rec_mean = self.dataset['mean_r_end'][self.r]
        end_rec_var = self.dataset['var_r_end'][self.r]

        id_year_map = self.dataset['id2time']

        tstart_min, tstart_max, tend_min, tend_max = self.get_start_end_ranges(allen_pred, e1, e2)

        net_sum_start = 0.0
        net_sum_end = 0.0

        for tstart in range(tstart_min, tstart_max):
            if (tstart!=self.mintime and tstart!=self.maxtime):
                tstart_act = int(id_year_map[tstart])
                e1_act = int(id_year_map[e1])
                # map_answer_start should be dependent on the rule head
                if ((tstart_act - e1_act) not in self.map_answer_start[normal_pred]):
                    diff = tstart_act-e1_act
                    x = -(start_mean_diff - diff)**2
                    x = x/(2*start_var_diff)

                    if (self.r == normal_pred):
                    
                        diff_rec = abs(tstart_act - e1_act)
                        
                        x = - (start_rec_mean - diff_rec)**2
                        x = x/(2*start_rec_var)

                    self.map_answer_start[normal_pred][tstart_act - e1_act] = torch.exp(x).item()

                map_answer_start_local[tstart] =self.map_answer_start[normal_pred][tstart_act-e1_act]

                net_sum_start += map_answer_start_local[tstart]
  


        for tend in range(tend_min, tend_max):
            if (tend!=self.mintime and tend!=self.maxtime):
                tend_act = int(id_year_map[tend])
                e2_act = int(id_year_map[e2])
                
                if ((tend_act - e2_act) not in self.map_answer_end[normal_pred]):
                    diff = tend_act-e2_act
                    x = -(end_mean_diff - diff)**2
                    x = x/(2*end_var_diff)


                    if (self.r == normal_pred):
                 
                        diff_rec = abs(tend_act - e2_act)
                        x = - (end_rec_mean - diff_rec)**2
                        x = x/(2*end_rec_var)

                    self.map_answer_end[normal_pred][tend_act - e2_act] = torch.exp(x).item()
                    
                map_answer_end_local[tend] =  self.map_answer_end[normal_pred][tend_act-e2_act]

                net_sum_end += map_answer_end_local[tend]

        if (net_sum_start > 0):
            for key,value in map_answer_start_local.items():  
                map_answer_start_local[key] = (value/net_sum_start)*count

        if (net_sum_end > 0):
            for key,value in map_answer_end_local.items():
                map_answer_end_local[key] = (value/net_sum_end)*count

        return map_answer_start_local, map_answer_end_local 

    def link_groundings(self,h,rule,graph,link_answer,remove_edges): 

        t_list_set = set(link_answer)
        m = [dict() for i in range(int(len(rule)+1))]
        m[0][h] = 1
        map_answer = dict()
    
        for j in range(int(len(rule))):
            for elem,count in m[j].items():
                curr_s = elem[0]
                curr_t1 = elem[1]
                curr_t2 = elem[2]
                curr_r = rule[j]%(self.R)
                curr_allen_r = int(rule[j]/(self.R))

                for k in range(len(graph[curr_s][curr_r])): 
                    
                    next_s = graph[curr_s][curr_r][k][0]
                    next_t1 = graph[curr_s][curr_r][k][1]
                    next_t2 = graph[curr_s][curr_r][k][2]
                  
                    if (self.r == curr_r and next_t1 == h[1] and next_t2 == h[2] and remove_edges):
                        if (curr_s== h[0] and (next_s in t_list_set)): 

                            continue

                    if (curr_allen_r !=13): 
                        if (self.allen_relation(curr_t1,curr_t2,next_t1,next_t2) != curr_allen_r):
                            continue # 

 
                    if ((next_s,next_t1,next_t2) not in m[j+1].keys()):
                        m[j+1][(next_s,next_t1,next_t2)]=0
                    m[j+1][(next_s,next_t1,next_t2)] =  m[j+1][(next_s,next_t1,next_t2)] + count
               
        for elem,count in m[int(len(rule))].items():
            curr_s = elem[0]
            curr_t1 = elem[1]
            curr_t2 = elem[2]
            if (curr_s not in map_answer.keys()):
                map_answer[curr_s] =0
            map_answer[curr_s] += count

        rgnd = []
        rgnd_count = []
   
        for elem,count in map_answer.items():
            rgnd.append(elem)
            rgnd_count.append(count) 

        return rgnd,rgnd_count

    def time_groundings(self,h,rule,graph):
       
        e1 = ''
        e2 = ''
               
        
        rgnd_start = []
        rgnd_start_count = []
        rgnd_end = []
        rgnd_end_count = []
        

        target_allen_predicate = -1
        m = [dict() for i in range(int(len(rule)+1))]
        m[0][h] = 1 # 1 represents count

        map_answer_start = dict()
        map_answer_end = dict()
        
        E_map = dict()
        E_starts = dict()
        for k in range(len(graph[h[0]][rule[0]%(self.R)])):

            next_s = graph[h[0]][rule[0]%(self.R)][k][0]
            E_s = graph[h[0]][rule[0]%(self.R)][k][1]  #next_t1
            E_e = graph[h[0]][rule[0]%(self.R)][k][2]  #next_t2
          
            if ((E_s,E_e) not in E_map.keys()):
                E_map[(E_s,E_e)] = [dict() for i in range(int(len(rule)))]
            E_map[(E_s,E_e)][0][(next_s,E_s,E_e)] = 1 

        target_allen_predicate  = int(rule[0]/(self.R))
        target_normal_predicate = int(rule[0]%(self.R))
        for starting_E in E_map.keys():
            e1 = starting_E[0]
            e2 = starting_E[1]
            for j in range(1,int(len(rule))):
                for elem,count in E_map[starting_E][j-1].items():
                    curr_s = elem[0]
                    curr_t1 = elem[1]
                    curr_t2 = elem[2]
                    curr_r = rule[j]%(self.R)
                    curr_allen_r = int(rule[j]/(self.R))
                    for k in range(len(graph[curr_s][curr_r])): 
                        next_s = graph[curr_s][curr_r][k][0]
                        next_t1 = graph[curr_s][curr_r][k][1]
                        next_t2 = graph[curr_s][curr_r][k][2]

                        if (self.allen_relation(curr_t1,curr_t2,next_t1,next_t2) != curr_allen_r):
                            continue # Violation of allen constraint

                        if ((next_s,next_t1,next_t2) not in E_map[starting_E][j].keys()):
                            E_map[starting_E][j][(next_s,next_t1,next_t2)]=0
                        E_map[starting_E][j][(next_s,next_t1,next_t2)] = E_map[starting_E][j][(next_s,next_t1,next_t2)] + count
            
            for elem,count in E_map[starting_E][int(len(rule)) - 1].items():
                curr_s = elem[0]
                curr_t1 = elem[1]
                curr_t2 = elem[2]
                frequency_sum = 0

                if (curr_s == h[1]): # This is a valid grounding
                    # for e1,e2 as start, count will  be multiplied to each frquency
                    if (starting_E[0] not in E_starts.keys()):
                        E_starts[starting_E[0]] = 0
                    E_starts[starting_E[0]] += count
                    
                    map_answer_start1, map_answer_end1 = self.potential_distribution_for_end_points(target_allen_predicate, target_normal_predicate, count, e1, e2)
                                            
                    for tinstance,frequency in map_answer_start1.items():
                        rgnd_start.append(tinstance) 
                        rgnd_start_count.append(frequency) 

                    for tinstance,frequency in map_answer_end1.items():
                        rgnd_end.append(tinstance)
                        rgnd_end_count.append(frequency)
          
        return rgnd_start, rgnd_start_count, rgnd_end, rgnd_end_count

    def PCA_link(self,batch):
        num_rule = self.num_rule
        element, link_answer, link_t_list, time_answer, time_t_list, masked_link, crule_link, crule_start, crule_end, centity, cstart, cend, ccount_link, ccount_start, ccount_end =self.load_batch(batch)
        # Here,
        with torch.no_grad():

            cscore = torch.ones_like(centity)
            indices = torch.stack([crule_link, centity], 0)

            def cvalue(cscore):
                if cscore.size(0) == 0:
                    return torch.zeros(num_rule).cuda()
                return torch.sparse.sum(torch.sparse.FloatTensor( # Making float tensor non-zero at these crule and centity indices
                    indices,
                    cscore,
                    torch.Size([num_rule, self.E])
                ).cuda(), -1).to_dense()
   

            pos = cvalue(cscore * masked_link[centity])
            neg = cvalue(cscore * ~masked_link[centity])

            self.prec_num_link += pos # maintain numerators and denominators separately
            self.prec_denom_link += (pos+neg)

        return torch.true_divide(pos,(pos+neg))

    def PCA_temporal(self,batch):
        # SEE THIS FUNCTION
        num_rule = self.num_rule
        element, link_answer, link_t_list, time_answer, time_t_list, masked_link, crule_link, crule_start, crule_end, centity, cstart, cend, ccount_link, ccount_start, ccount_end =self.load_batch(batch)
        with torch.no_grad():

            def Extract(lst, i):
                return [item[i] for item in lst]

            mask_start = list2mask(Extract(time_t_list, 0),  self.totalintervals).cuda()
            mask_end = list2mask(Extract(time_t_list, 1),  self.totalintervals).cuda() 

       
            cscore_start = torch.ones_like(cstart) # grounded starts # this is all ones
            indices_start = torch.stack([crule_start, cstart], 0)

            cscore_end = torch.ones_like(cend)
            indices_end = torch.stack([crule_end, cend], 0)


            def cvalue(cscore, indices):
                if cscore.size(0) == 0:
                    return torch.zeros(num_rule).cuda()
                return torch.sparse.sum(torch.sparse.FloatTensor( # Making float tensor non-zero at these crule and centity indices
                    indices,
                    cscore,
                    torch.Size([num_rule, self.totalintervals])
                ).cuda(), -1).to_dense()
            
            # Rule precision will simply be pos/(pos+neg), cweight is 1
            pos_start = cvalue(cscore_start * mask_start[cstart], indices_start).clamp(min=0.0001)

            pos_end = cvalue(cscore_end * mask_end[cend], indices_end).clamp(min=0.0001)

            neg_start = cvalue(cscore_start * ~mask_start[cstart], indices_start).clamp(min=0.0001)
            neg_end = cvalue(cscore_end * ~mask_end[cend], indices_end).clamp(min=0.0001)

            self.prec_num_start += pos_start # maintain numerators and denominators separately
            self.prec_denom_start += (pos_start+neg_start)

            self.prec_num_end +=pos_end # maintain numerators and denominators separately
            self.prec_denom_end +=(pos_end+neg_end)

        return torch.true_divide(pos_start,(pos_start+neg_start)), torch.true_divide(pos_end,(pos_end+neg_end))


    def relation_init(self, r=None, rule_file=None, rules=None):
        print = self.print
        if r is not None:
            self.r = r

        r = self.r
        if rules is None:
            assert rule_file is not None
            prior_to_find = True

            rules = []
            rule_set = set()
            rules = [(((self.R-1)*(12) +r,), 1,1e9, -1)] # The trivial rule will be equals, r -> 12*20 + r
            rule_set = set([tuple(), ((self.R-1)*(12) +r,)])
            with open(rule_file) as file:
                for i, line in enumerate(file):
                    #print(line)
                    try:
                        path, prec, num_gr = line.split('\t')
                        path = tuple(map(int, path.split()))
                        prec = float(prec.split()[0])
                        num_gr = int(num_gr.split()[0])

                        if path in rule_set:
                            continue
                    
                        rule_set.add(path)
                        if len(path) <= self.arg('max_rule_len'):
                            rules.append((path, prec, num_gr, i))
                    except:
                        continue

            rules = sorted(rules, key=lambda x: (x[1], x[2]), reverse=True)
            print(f"Loaded from file: |rules| = {len(rules)} max_rule_len = {self.arg('max_rule_len')}")
            x = torch.tensor([prec for _, prec,num_gr, _ in rules]).cuda() 
            prior = x
            rules = [path for path, _, _,_ in rules]
        else:
            assert prior is not None

        self.prior_link = prior
        self.prior_start = prior
        self.prior_end = prior
        self.set_rules(rules)

        num_rule = self.num_rule
        with torch.no_grad():
            self.prec_num_link = torch.zeros(num_rule).cuda() 
            self.prec_denom_link = torch.zeros(num_rule).cuda()
            self.prec_num_start = torch.zeros(num_rule).cuda() 
            self.prec_denom_start = torch.zeros(num_rule).cuda() 
            self.prec_num_end = torch.zeros(num_rule).cuda() 
            self.prec_denom_end = torch.zeros(num_rule).cuda() 
  
        if (prior_to_find):
            for batch in self.make_batchs(init=True): #
                if (self.train_mode!='time'):
                    self.PCA_link(batch)
                if (self.train_mode!='link'):
                    self.PCA_temporal(batch)
            

            self.prior_link = torch.true_divide(self.prec_num_link,self.prec_denom_link)
            self.prior_link[torch.isnan(self.prior_link)] = 0 # Handling Nan's
            self.prior_link[0] = 1.0 # trivial rule

            self.prior_start = torch.true_divide(self.prec_num_start,self.prec_denom_start)
            self.prior_start[torch.isnan(self.prior_start)] = 0
            self.prior_start[0] = 1.0 # trivial rule

            self.prior_end = torch.true_divide(self.prec_num_end,self.prec_denom_end )
            self.prior_end[torch.isnan(self.prior_end)] = 0
            self.prior_end[0] = 1.0 # trivial rule

        self.eta_learnable = torch.nn.Parameter(torch.ones(1)*self.eta) 

        self.make_batchs()

    def set_args(self,lr,pe,max_rule_len):
        self._args = dict()
        def_args = dict()
        
        def_args['max_rule_len'] = max_rule_len
        def_args['predictor_num_epoch'] = pe 
        def_args['predictor_lr'] = lr
        def_args['predictor_batch_size'] = 1
        def_args['predictor_print_epoch'] = 50
        def_args['predictor_init_print_epoch'] = 10
        def_args['predictor_valid_epoch'] = 100
        def_args['predictor_eval_rate'] = 4
        def_args['rule_value_def'] = '(pos - neg) / num'
        def_args['metrics_score_def'] = '(mrr)'#for link
        def_args['answer_candidates'] = None
        def_args['record_test'] = False

        def make_str(v):
            if isinstance(v, int):
                return True
            if isinstance(v, float):
                return True
            if isinstance(v, bool):
                return True
            if isinstance(v, str):
                return True
            return False

        for k, v in def_args.items():
            self._args[k] = str(v) if make_str(v) else v

    def arg(self, name, apply=None):
        # print(self._args[name])
        v = self._args[name]
        if apply is None:
            if v is None:
                return None
            return eval(v)
        return apply(v)
    
    def get_knowns(self,h,r,t,knowns):

        ks = []
        for i in range(h.shape[0]):
            ks.append(knowns[(h[i].cpu().item(),r[i].cpu().item(),t[i].cpu().item())])

        lens = [len(x) for x in ks]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens - len(x)), 'edge')
              for x in ks]
        result = numpy.array(ks)
        return result

    def _make_batch(self, element,graph, link_answer=None, link_t_list = None, time_answer=None, time_t_list=None, type_batch='both', rgnd_buffer=None, remove_edges = False):
   
        if rgnd_buffer is None: # Training/ Validation Time
            rgnd_buffer_link = self.rgnd_buffer_link
            rgnd_buffer_time = self.rgnd_buffer_time
        else:
            if (type_batch == 'link'):
                rgnd_buffer_link = rgnd_buffer
            else:
                rgnd_buffer_time = rgnd_buffer

        crule_link = []
        crule_start = []
        crule_end = []

        centity = []
        cstart = []
        cend = []

        ccount_link = []
        ccount_start = []
        ccount_end = []
       
        if (type_batch =='link'):
            for i, rule in enumerate(self.rules_exp):
                if i != 0:
                    key_link = (element[0], self.r, element[2], element[3], rule)

                    if (key_link in rgnd_buffer_link): # Buffer to optimize computation
                        rgnd, rgnd_count = rgnd_buffer_link[key_link]

                    else:

                        rgnd, rgnd_count = self.link_groundings((element[0],element[2],element[3]), rule, graph, link_answer, remove_edges)
                        rgnd_buffer_link[key_link] = (rgnd, rgnd_count)

                    ones = torch.ones(len(rgnd))
                    centity.append(torch.LongTensor(rgnd))
                    ccount_link.append(torch.LongTensor(rgnd_count))
                    crule_link.append(ones.long() * i) # crule is repeated centity times
     
                else:
                    rgnd = []
                    rgnd_count = []

        if (type_batch =='time'):
            for i, rule in enumerate(self.rules_exp):
                if i != 0:
                    key_time = (element[0], self.r, element[1], rule)

                    if (key_time in rgnd_buffer_time): # Buffer to optimize computation
                        rgnd_start, rgnd_start_count, rgnd_end, rgnd_end_count = rgnd_buffer_time[key_time]

                    else:
                    
                        rgnd_start, rgnd_start_count, rgnd_end, rgnd_end_count = self.time_groundings((element[0],element[1]), rule, graph)
                        rgnd_buffer_time[key_time] = (rgnd_start, rgnd_start_count, rgnd_end, rgnd_end_count)


                    ones_start = torch.ones(len(rgnd_start))
                    ones_end = torch.ones(len(rgnd_end))

                    cstart.append(torch.LongTensor(rgnd_start)) # tinstance
                    ccount_start.append(torch.FloatTensor(rgnd_start_count)) # distribution

                    cend.append(torch.LongTensor(rgnd_end)) # tinstance
                    ccount_end.append(torch.FloatTensor(rgnd_end_count)) # distribution

                    crule_start.append(ones_start.long() * i)
                    crule_end.append(ones_end.long() * i)

                else:
                    
                    rgnd_start = []
                    rgnd_end = []
                    rgnd_start_count = []
                    rgnd_end_count = []
           

        # print("iter done")
        if (len(crule_link) == 0):
            crule_link = torch.tensor([]).long().cuda()
            centity = torch.tensor([]).long().cuda()
            ccount_link = torch.tensor([]).long().cuda()
        else:
            crule_link = torch.cat(crule_link, dim=0)
            centity = torch.cat(centity, dim=0)
            ccount_link = torch.cat(ccount_link, dim=0)

        if (len(crule_start) == 0):
            crule_start = torch.tensor([]).long().cuda()
            cstart = torch.tensor([]).long().cuda()
            ccount_start = torch.tensor([]).float().cuda()
        else:
            crule_start = torch.cat(crule_start, dim=0)
            cstart = torch.cat(cstart, dim=0)
            ccount_start = torch.cat(ccount_start, dim=0)

        if (len(crule_end) == 0):
            crule_end = torch.tensor([]).long().cuda()
            cend = torch.tensor([]).long().cuda()
            ccount_end = torch.tensor([]).float().cuda()
        else:
            crule_end = torch.cat(crule_end, dim=0)
            cend = torch.cat(cend, dim=0)
            ccount_end = torch.cat(ccount_end, dim=0)

        masked_entities = None
        if (link_answer!=None):
            masked_entities =  list2mask(link_answer, self.E)
        return element, link_answer, link_t_list, time_answer, time_t_list, masked_entities, crule_link, crule_start, crule_end, centity, cstart, cend, ccount_link, ccount_start, ccount_end

    def make_batchs(self, init=False):
        print = self.print

        dataset = self.dataset
        graph = build_graph(dataset['train'], self.E, self.R) # Graph used for training
        bg_graph = build_graph(dataset['bg'], self.E, self.R) # background kg used for grounding



        def filter(tri): 
            a = []
            for h, r, t, t1, t2 in tri:
                if r == self.r:
                    a.append((h,t,t1,t2)) 
            return a

        def filter_valid_test(tri):
            a = defaultdict(lambda:[])
            for h, r, t, t1, t2 in tri:
                if r == self.r:
                    a[(h,t1,t2)].append(t)
            return a

        def filter_valid_test_time(tri):
            a = defaultdict(lambda:[])
            cnt = 0
            for h, r, t, t1, t2 in tri:
                if r == self.r:
                    a[(h,t)].append([t1,t2])
                    cnt += 1
            return a,cnt



        train = filter(dataset['train'])
       
        if (self.train_mode == 'link'):
            valid = filter_valid_test(dataset['valid'])
            test = filter_valid_test(dataset['test'])
            answer_valid_link = defaultdict(lambda: [])
            answer_test_link  = defaultdict(lambda: [])
        else:
            valid_time, cnt_valid_time = filter_valid_test_time(dataset['valid'])  
            test_time, cnt_test_time = filter_valid_test_time(dataset['test'])
            t_list_valid_time = defaultdict(lambda:[])
            t_list_test_time = defaultdict(lambda:[])
            answer_valid_time = defaultdict(lambda: [])
            answer_test_time  = defaultdict(lambda: [])
            kb_valid_time = self.kvalid_sub.facts # FOR TIME
            kb_test_time  = self.ktest_sub.facts # FOR TIME
            for onefact in kb_valid_time:
                if onefact[1] == self.r:
                    answer_valid_time[tuple(onefact[:3])].append(onefact[3:].tolist())

            for onefact in kb_test_time:
                if onefact[1] == self.r:
                    answer_test_time[tuple(onefact[:3])].append(onefact[3:].tolist())
        

        answer_train_link = defaultdict(lambda: [])
        answer_train_time = defaultdict(lambda: [])
        t_list_train_time = defaultdict(lambda:[])
       
           
        for elems in train:
            answer_train_time[tuple([elems[0],self.r,elems[1]])] = [elems[2], elems[2], elems[3], elems[3]]
        
        

        for elements in train:
            # h,t,t1,t2
            k = (elements[0],elements[2],elements[3])
            v = elements[1]
            k_time = (elements[0],elements[1])

            t_list_train_time[k_time].append([elements[2],elements[3]])

            answer_train_link[k].append(v)
            if (self.train_mode == 'link'):
                answer_valid_link[k].append(v)
                answer_test_link[k].append(v)
            
        if (self.train_mode == 'link'):
            for k,v in valid.items():
                answer_valid_link[k] += v
                answer_test_link[k] += v

            for k,v in test.items():

                answer_test_link[k] += v


        print_epoch = self.arg('predictor_init_print_epoch')

        self.train_batch = []
        self.valid_batch = []
        self.test_batch = []
        self.valid_batch_time = []
        self.test_batch_time = [] 

        if init:
            def gen_init(self, train, print_epoch):
                for i, elements in enumerate(train):
                    if i % print_epoch == 0:
                        print(f"init_batch: {i}/{len(train)}")
            
                    yield self._make_batch(elements, graph, link_answer = answer_train_link[(elements[0],elements[2],elements[3])],link_t_list = answer_train_link[(elements[0],elements[2],elements[3])], time_answer = answer_train_time[tuple([elements[0],self.r ,elements[1]])], time_t_list = t_list_train_time[(elements[0],elements[1])], type_batch = self.train_mode, remove_edges=True)

            return gen_init(self, train, print_epoch)

        for i, elements in enumerate(train):
            if i % print_epoch == 0:
                print(f"train_batch: {i}/{len(train)}")

            batch = list(self._make_batch(elements, graph, link_answer = answer_train_link[(elements[0],elements[2],elements[3])], link_t_list =answer_train_link[(elements[0],elements[2],elements[3])], time_answer = answer_train_time[tuple([elements[0],self.r ,elements[1]])],time_t_list = t_list_train_time[(elements[0],elements[1])], type_batch = self.train_mode, remove_edges=True))
    
            self.train_batch.append(tuple(batch))

        if (self.train_mode == 'link'):
            for i, (h,t_list) in enumerate(valid.items()):
            
                if i % print_epoch == 0:
                    print(f"valid_batch Link: {i}/{len(valid)}")

                elements = (h[0], t_list[0], h[1], h[2]) 
                self.valid_batch.append(self._make_batch(elements, graph, link_answer = answer_valid_link[(elements[0],elements[2],elements[3])], link_t_list = t_list, type_batch= 'link'))

        if (self.train_mode == 'time'):
           for i, (h,t_list) in enumerate(valid_time.items()):
              
                if i % print_epoch == 0:
                    print(f"valid_batch Time: {i}/{cnt_valid_time}")

                elements = (h[0], h[1], t_list[0][0], t_list[0][1])
                batch_made = list(self._make_batch(elements, graph, time_answer = answer_valid_time[tuple([elements[0],self.r ,elements[1]])], time_t_list =t_list,type_batch= 'time'))
                for i in range(len(answer_valid_time[tuple([elements[0],self.r ,elements[1]])])):
                    batch_made[3] = answer_valid_time[tuple([elements[0],self.r ,elements[1]])][i]
                    self.valid_batch_time.append(batch_made)


        if (self.train_mode=='link'):
            for i, (h,t_list) in enumerate(test.items()):
                if i % print_epoch == 0:
                    print(f"test_batch Link: {i}/{len(test)}")

                elements = (h[0], t_list[0], h[1], h[2])
                self.test_batch.append(self._make_batch(elements, bg_graph, link_answer = answer_test_link[(elements[0],elements[2],elements[3])], link_t_list = t_list, type_batch= 'link', rgnd_buffer=self.rgnd_buffer_test_link))

        if (self.train_mode=='time'):

            for i, (h,t_list) in enumerate(test_time.items()):
              
                if i % print_epoch == 0:
                    print(f"test_batch Time: {i}/{cnt_test_time}")

                elements = (h[0], h[1], t_list[0][0], t_list[0][1])
                batch_made = list(self._make_batch(elements, bg_graph, time_answer = answer_test_time[tuple([elements[0],self.r ,elements[1]])], time_t_list =t_list,type_batch= 'time', rgnd_buffer=self.rgnd_buffer_test_time))
                for i in range(len(answer_test_time[tuple([elements[0],self.r ,elements[1]])])):
                    batch_made[3] = answer_test_time[tuple([elements[0],self.r ,elements[1]])][i]
                    self.test_batch_time.append(batch_made)
    



