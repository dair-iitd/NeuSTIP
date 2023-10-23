import os
import torch
import numpy as np
import json
time_index = {"t_s": 0, "t_s_orig": 1, "t_e": 2, "t_e_orig": 3, "t_str": 4, "t_i": 5}
class RotatE(torch.nn.Module):
    def __init__(self, path):
        super(RotatE, self).__init__()
        self.path = path

        cfg_file = os.path.join(path, 'config.json')
        with open(cfg_file, 'r') as fi:
            cfg = json.load(fi)
        self.emb_dim = cfg['hidden_dim']
        self.gamma = cfg['gamma']
        self.range = (self.gamma + 2.0) / self.emb_dim
        self.num_entities = cfg['nentity']

        eemb_file = os.path.join(path, 'entity_embedding.npy')
        eemb = np.load(eemb_file)
        self.eemb = torch.nn.parameter.Parameter(torch.tensor(eemb))

        remb_file = os.path.join(path, 'relation_embedding.npy')
        remb = np.load(remb_file)
        remb = torch.tensor(remb)
        self.remb = torch.nn.parameter.Parameter(torch.cat([remb, -remb], dim=0))

    def product(self, vec1, vec2):
        re_1, im_1 = torch.chunk(vec1, 2, dim=-1)
        re_2, im_2 = torch.chunk(vec2, 2, dim=-1)

        re_res = re_1 * re_2 - im_1 * im_2
        im_res = re_1 * im_2 + im_1 * re_2

        return torch.cat([re_res, im_res], dim=-1)

    def project(self, vec):
        pi = 3.141592653589793238462643383279
        vec = vec / (self.range / pi)

        re_r = torch.cos(vec)
        im_r = torch.sin(vec)

        return torch.cat([re_r, im_r], dim=-1)

    def diff(self, vec1, vec2):
        diff = vec1 - vec2
        re_diff, im_diff = torch.chunk(diff, 2, dim=-1)
        diff = torch.stack([re_diff, im_diff], dim=0)
        diff = diff.norm(dim=0)
        return diff

    def dist(self, all_h, all_r, all_t):
        h_emb = self.eemb.index_select(0, all_h).squeeze()
        r_emb = self.remb.index_select(0, all_r).squeeze()
        t_emb = self.eemb.index_select(0, all_t).squeeze()

        r_emb = self.project(r_emb)
        e_emb = self.product(h_emb, r_emb)
        dist = self.diff(e_emb, t_emb)

        return dist.sum(dim=-1)
    
    def forward(self, all_h, all_r):
        all_h_ = all_h.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_r_ = all_r.unsqueeze(-1).expand(-1, self.num_entities).reshape(-1)
        all_e_ = torch.tensor(list(range(self.num_entities)), dtype=torch.long, device=all_h.device).unsqueeze(0).expand(all_r.size(0), -1).reshape(-1)
        kge_score = self.gamma - self.dist(all_h_, all_r_, all_e_)
        kge_score = kge_score.view(-1, self.num_entities)
        return kge_score

class TimePlex(torch.nn.Module):
    def __init__(self, path, embedding_dim, srt_wt, ort_wt, sot_wt, nentity, nrelation, pairs_wt, recurrent_wt, use_inverse, id_to_year):
        super(TimePlex, self).__init__()
        self.path = path

        E_im_file = os.path.join(path,'timeplex/E_im.npy') 
        E_im = np.load(E_im_file)
        self.E_im = torch.nn.parameter.Parameter(torch.tensor(E_im))
        
        E_re_file = os.path.join(path,'timeplex/E_re.npy') 
        E_re = np.load(E_re_file)
        self.E_re = torch.nn.parameter.Parameter(torch.tensor(E_re))

        R_im_file = os.path.join(path,'timeplex/R_im.npy') 
        R_im = np.load(R_im_file)
        self.R_im = torch.nn.parameter.Parameter(torch.tensor(R_im))

        R_re_file = os.path.join(path,'timeplex/R_re.npy') 
        R_re = np.load(R_re_file)
        self.R_re = torch.nn.parameter.Parameter(torch.tensor(R_re))
        
        E2_im_file = os.path.join(path,'timeplex/E2_im.npy') 
        E2_im = np.load(E2_im_file)
        self.E2_im = torch.nn.parameter.Parameter(torch.tensor(E2_im))
        
        E2_re_file = os.path.join(path,'timeplex/E2_re.npy') 
        E2_re = np.load(E2_re_file)
        self.E2_re = torch.nn.parameter.Parameter(torch.tensor(E2_re))
        
        Rs_im_file = os.path.join(path,'timeplex/Rs_im.npy') 
        Rs_im = np.load(Rs_im_file)
        self.Rs_im = torch.nn.parameter.Parameter(torch.tensor(Rs_im))

        Rs_re_file = os.path.join(path,'timeplex/Rs_re.npy') 
        Rs_re = np.load(Rs_re_file)
        self.Rs_re = torch.nn.parameter.Parameter(torch.tensor(Rs_re))
               
        Ro_im_file = os.path.join(path,'timeplex/Ro_im.npy') 
        Ro_im = np.load(Ro_im_file)
        self.Ro_im = torch.nn.parameter.Parameter(torch.tensor(Ro_im))

        Ro_re_file = os.path.join(path,'timeplex/Ro_re.npy') 
        Ro_re = np.load(Ro_re_file)
        self.Ro_re = torch.nn.parameter.Parameter(torch.tensor(Ro_re))

        Ts_im_file = os.path.join(path,'timeplex/Ts_im.npy') 
        Ts_im = np.load(Ts_im_file)
        self.Ts_im = torch.nn.parameter.Parameter(torch.tensor(Ts_im))

        Ts_re_file = os.path.join(path,'timeplex/Ts_re.npy') 
        Ts_re = np.load(Ts_re_file)
        self.Ts_re = torch.nn.parameter.Parameter(torch.tensor(Ts_re))
               
        To_im_file = os.path.join(path,'timeplex/To_im.npy') 
        To_im = np.load(To_im_file)
        self.To_im = torch.nn.parameter.Parameter(torch.tensor(To_im))

        To_re_file = os.path.join(path,'timeplex/To_re.npy') 
        To_re = np.load(To_re_file)
        self.To_re = torch.nn.parameter.Parameter(torch.tensor(To_re))  
         
        t1_emb_file = os.path.join(path,'timeplex/t1_emb.npy') 
        t1_emb = np.load(t1_emb_file)
        self.t1_emb = torch.nn.parameter.Parameter(torch.tensor(t1_emb))  

    

        sub_degree_file = os.path.join(path,'timeplex/sub_degree.npy') 
        sub_degree = np.load(sub_degree_file)
        self.sub_degree = torch.nn.parameter.Parameter(torch.tensor(sub_degree))

        obj_degree_file = os.path.join(path,'timeplex/obj_degree.npy') 
        obj_degree = np.load(obj_degree_file)
        self.obj_degree = torch.nn.parameter.Parameter(torch.tensor(obj_degree))


        mean_r_sub_file = os.path.join(path,'timeplex/mean_r_sub.npy') 
        mean_r_sub = np.load(mean_r_sub_file)
        self.mean_r_sub = torch.nn.parameter.Parameter(torch.tensor(mean_r_sub))

        mean_r_obj_file = os.path.join(path,'timeplex/mean_r_obj.npy') 
        mean_r_obj = np.load(mean_r_obj_file)
        self.mean_r_obj = torch.nn.parameter.Parameter(torch.tensor(mean_r_obj))

        var_r_sub_file = os.path.join(path,'timeplex/var_r_sub.npy') 
        var_r_sub = np.load(var_r_sub_file)
        self.var_r_sub = torch.nn.parameter.Parameter(torch.tensor(var_r_sub))

        var_r_obj_file = os.path.join(path,'timeplex/var_r_obj.npy') 
        var_r_obj = np.load(var_r_obj_file)
        self.var_r_obj = torch.nn.parameter.Parameter(torch.tensor(var_r_obj))

        offset_r_sub_file = os.path.join(path,'timeplex/offset_r_sub.npy') 
        offset_r_sub = np.load(offset_r_sub_file)
        self.offset_r_sub = torch.nn.parameter.Parameter(torch.tensor(offset_r_sub))

        offset_r_obj_file = os.path.join(path,'timeplex/offset_r_obj.npy') 
        offset_r_obj = np.load(offset_r_obj_file)
        self.offset_r_obj = torch.nn.parameter.Parameter(torch.tensor(offset_r_obj))

        W_r_sub_file = os.path.join(path,'timeplex/W_r_sub.npy') 
        W_r_sub = np.load(W_r_sub_file)
        self.W_r_sub = torch.nn.parameter.Parameter(torch.tensor(W_r_sub))

        W_r_obj_file = os.path.join(path,'timeplex/W_r_obj.npy') 
        W_r_obj = np.load(W_r_obj_file)
        self.W_r_obj = torch.nn.parameter.Parameter(torch.tensor(W_r_obj))

        mean_r_r_sub_file = os.path.join(path,'timeplex/mean_r_r_sub.npy') 
        mean_r_r_sub = np.load(mean_r_r_sub_file)
        self.mean_r_r_sub = torch.nn.parameter.Parameter(torch.tensor(mean_r_r_sub))

        mean_r_r_obj_file = os.path.join(path,'timeplex/mean_r_r_obj.npy') 
        mean_r_r_obj = np.load(mean_r_r_obj_file)
        self.mean_r_r_obj = torch.nn.parameter.Parameter(torch.tensor(mean_r_r_obj))

        var_r_r_sub_file = os.path.join(path,'timeplex/var_r_r_sub.npy') 
        var_r_r_sub = np.load(var_r_r_sub_file)
        self.var_r_r_sub = torch.nn.parameter.Parameter(torch.tensor(var_r_r_sub))

        var_r_r_obj_file = os.path.join(path,'timeplex/var_r_r_obj.npy') 
        var_r_r_obj = np.load(var_r_r_obj_file)
        self.var_r_r_obj = torch.nn.parameter.Parameter(torch.tensor(var_r_r_obj))

        offset_r_r_sub_file = os.path.join(path,'timeplex/offset_r_r_sub.npy') 
        offset_r_r_sub = np.load(offset_r_r_sub_file)
        self.offset_r_r_sub = torch.nn.parameter.Parameter(torch.tensor(offset_r_r_sub))

        offset_r_r_obj_file = os.path.join(path,'timeplex/offset_r_r_obj.npy') 
        offset_r_r_obj = np.load(offset_r_r_obj_file)
        self.offset_r_r_obj = torch.nn.parameter.Parameter(torch.tensor(offset_r_r_obj))

        W_sub_file = os.path.join(path,'timeplex/W_sub.npy') 
        W_sub = np.load(W_sub_file)
        self.W_sub = torch.nn.parameter.Parameter(torch.tensor(W_sub))

        W_obj_file = os.path.join(path,'timeplex/W_obj.npy') 
        W_obj = np.load(W_obj_file)
        self.W_obj = torch.nn.parameter.Parameter(torch.tensor(W_obj))
        
        self.sub_facts  = torch.load(os.path.join(path,'timeplex/sub_facts'))
        self.obj_facts = torch.load(os.path.join(path,'timeplex/obj_facts'))

        self.embedding_dim = embedding_dim
        self.srt_wt = srt_wt
        self.ort_wt = ort_wt
        self.sot_wt = sot_wt
        self.num_entities = nentity
        self.pairs_wt = pairs_wt
        self.recurrent_wt = recurrent_wt
        self.relation_count = nrelation+1 
        self.weight_init = 0.1
       
        self.mask_r_r = torch.ones(self.relation_count, self.relation_count).cuda()

  
        self.mask_r_r[self.relation_count-1, :] = 0
        self.mask_r_r[:, self.relation_count-1] = 0
        self.use_obj_scores = True



        self.t2_emb = torch.nn.Embedding(len(id_to_year), 1)
        self.t2_emb.weight.requires_grad = False

        t_map = np.zeros((len(id_to_year), 1)) #len(id_to_year): 319
        self.dateYear2id = dict()
            
        for time_id in id_to_year:
            if (id_to_year[time_id]!='UNK-TIME'):
                t_map[time_id] = id_to_year[time_id]
                self.dateYear2id[int(id_to_year[time_id])] = time_id
            else:
                #t_map[time_id] = -1
                self.dateYear2id[id_to_year[time_id]] = time_id

        #self.t1_emb.weight.data.copy_(torch.from_numpy(t_map))
        self.t2_emb.weight.data.copy_(torch.from_numpy(t_map))
        if (not os.path.exists('t3_emb.npy')):
            np.save('t3_emb.npy',self.t2_emb.weight.detach().cpu().numpy())

        #self.eval_ids = torch.arange(len(id_to_year))
        self.eval_ids = torch.arange(self.num_entities)        
        self.eval_tensors = {'subject': self.get_nbors_indices(self.eval_ids, mode='subject'),
                             'object':  self.get_nbors_indices(self.eval_ids, mode='object')}

        self.eval_batch_size = 1
        self.use_inverse = use_inverse
        #self.predict_time = predict_time
        self.id_to_year = id_to_year

    @torch.no_grad()
    def convert_dict2mat(self,dict_in):
        dict_mat = np.zeros(len(dict_in))
        # ipdb.set_trace()
        try:
            for key in dict_in.keys():
                if type(key) == tuple:
                    if key == ('UNK-TIME', 'UNK-TIME'):
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(np.mean(key))
                else:
                    if key == 'UNK-TIME':
                        dict_mat[int(dict_in[key])] = -1
                    else:
                        dict_mat[int(dict_in[key])] = int(key)
        except:
            pdb.set_trace()
        dict_mat = np.array(dict_mat)
        return dict_mat

    @torch.no_grad()
    def complex_3way_fullsoftmax(self, s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, embedding_dim):
        #print("inside fullsoftmax s_re.shape: {} s_im.shape: {} r_re.shape: {} r_im.shape: {} o_re.shape: {} o_im.shape: {}".format(s_re.shape, s_im.shape, r_re.shape, r_im.shape, o_re.shape, o_im.shape)) #s_re.shape: torch.Size([24, 200]) s_im.shape: torch.Size([24, 200]) r_re.shape: torch.Size([24, 200]) r_im.shape: torch.Size([24, 200]) o_re.shape: torch.Size([24, 200]) o_im.shape: torch.Size([24, 200])
  
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im);  # tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re * r_re - s_im * r_im);  # tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not None:  # o.shape[1] > 1:
                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1,keepdim=True)
                del tmp1, tmp2
            else:  # all ent as neg samples
                tmp1 = tmp1.view(-1, embedding_dim)
                tmp2 = tmp2.view(-1, embedding_dim)

                o_re_tmp = o_re.view(-1, embedding_dim).transpose(0, 1)
                o_im_tmp = o_im.view(-1, embedding_dim).transpose(0, 1)
                #print(tmp1.shape,o_im_tmp.shape)
                result = tmp1 @ o_im_tmp + tmp2 @ o_re_tmp # IL x es
                del tmp1,o_im_tmp,tmp2,o_re_tmp
            # result.squeeze_()
        else:
            tmp1 = o_im * r_re - o_re * r_im;  # tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im * r_im + o_re * r_re;  # tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:  # s.shape[1] > 1:
                result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1,keepdim=True)
                del tmp1,tmp2
            else:
                tmp1 = tmp1.view(-1, embedding_dim)
                tmp2 = tmp2.view(-1, embedding_dim)

                s_im_tmp = s_im.view(-1, embedding_dim).transpose(0, 1)
                s_re_tmp = s_re.view(-1, embedding_dim).transpose(0, 1)
                result = tmp1 @ s_im_tmp + tmp2 @ s_re_tmp
                del tmp1, s_im_tmp,tmp2,s_re_tmp
            # result.squeeze_()
        return result
        
    @torch.no_grad() 
    def complex_3way_simple(self, s_re, s_im, r_re, r_im, o_re, o_im):  # <s,r,o_conjugate> when dim(s)==dim(r)==dim(o)
        #print("inside 3way simple s_re.shape: {} s_im.shape: {} r_re.shape: {} r_im.shape: {} o_re.shape: {} o_im.shape: {}".format(s_re.shape, s_im.shape, r_re.shape, r_im.shape, o_re.shape, o_im.shape))
        
        sro = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
        return sro.sum(dim=-1)    
    
    @torch.no_grad()
    def scoring_function(self,s,r,o,t,predict_time): 
        s_im = self.E_im.index_select(0,s.reshape(-1)).squeeze() if s is not None else self.E_im
        r_im = self.R_im.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.R_im
        o_im = self.E_im.index_select(0,o.reshape(-1)).squeeze() if o is not None else self.E_im
        s_re = self.E_re.index_select(0,s.reshape(-1)).squeeze() if s is not None else self.E_re
        r_re = self.R_re.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.R_re
        o_re = self.E_re.index_select(0,o.reshape(-1)).squeeze() if o is not None else self.E_re
        rs_im = self.Rs_im.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.Rs_im
        rs_re = self.Rs_re.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.Rs_re
        ro_im = self.Ro_im.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.Ro_im
        ro_re = self.Ro_re.index_select(0,r.reshape(-1)).squeeze() if r is not None else self.Ro_re
        t_re = self.Ts_re.index_select(0,t.reshape(-1)).squeeze() if t is not None else self.Ts_re
        t_im = self.Ts_im.index_select(0,t.reshape(-1)).squeeze() if t is not None else self.Ts_im
        t2_re = self.To_re.index_select(0,t.reshape(-1)).squeeze() if t is not None else self.To_re
        t2_im = self.To_im.index_select(0,t.reshape(-1)).squeeze() if t is not None else self.To_im
                
        if predict_time:
            
            s_re  = s_re.unsqueeze(0)
            s_im  = s_im.unsqueeze(0)
            rs_re = rs_re.unsqueeze(0)
            rs_im = rs_im.unsqueeze(0)
            ro_re = ro_re.unsqueeze(0)
            ro_im = ro_im.unsqueeze(0)
            o_re  = o_re.unsqueeze(0)
            o_im  = o_im.unsqueeze(0)
            
            srt = self.complex_3way_simple(s_re, s_im, rs_re, rs_im, t_re, t_im)
            
            # ort = complex_3way_simple(o_re, o_im, ro_re, ro_im, t_re, t_im)
            ort = self.complex_3way_simple(t_re, t_im, ro_re, ro_im, o_re, o_im)

            sot = self.complex_3way_simple(s_re, s_im,  t_re, t_im, o_re, o_im)

            score = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot    
            # --for inverse facts--#
            r = (r + self.relation_count - 1).to(torch.int32)
              
            rs_im = self.Rs_im.index_select(0,r.reshape(-1)).squeeze()
            rs_re = self.Rs_re.index_select(0,r.reshape(-1)).squeeze()
            ro_im = self.Ro_im.index_select(0,r.reshape(-1)).squeeze()
            ro_re = self.Ro_re.index_select(0,r.reshape(-1)).squeeze()
            
            srt = self.complex_3way_simple(o_re, o_im, rs_re, rs_im, t_re, t_im)
            ort = self.complex_3way_simple(t_re, t_im, ro_re, ro_im, s_re, s_im)
            sot = self.complex_3way_simple(o_re, o_im,  t_re, t_im, s_re, s_im)

            score_inv = self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot

            result = score + score_inv
            
            #print(" result.shape: {}".format(result.shape))
            
        else:
            sro = self.complex_3way_fullsoftmax(s, r, o, s_re, s_im, r_re, r_im, o_re, o_im, self.embedding_dim)
                
            srt = self.complex_3way_fullsoftmax(s, r, t, s_re, s_im, rs_re, rs_im, t_re, t_im, self.embedding_dim)
                
            ort = self.complex_3way_fullsoftmax(t, r, o, t_re, t_im, ro_re, ro_im, o_re, o_im, self.embedding_dim)

            sot = self.complex_3way_fullsoftmax(s, t, o, s_re, s_im, t_re, t_im, o_re, o_im,  self.embedding_dim)
            
            result = sro + self.srt_wt * srt + self.ort_wt * ort + self.sot_wt * sot
        
        #print(sro.shape,t.shape,srt.shape,ort.shape,sot.shape)
        del s_im,r_im,o_im,s_re,r_re,o_re,rs_im,rs_re,ro_im,ro_re,t_re,t_im,t2_re,t2_im
        return result
    @torch.no_grad()
    def get_nbors_indices(self, entities, mode='subject', filter = None):
        if mode == 'subject':
            nbor_dict = self.sub_facts
            degrees = self.sub_degree
        else:
            nbor_dict = self.obj_facts
            degrees = self.obj_degree

        nbors_to_filter = None
        if filter is not None:
            r, nbors, t = filter
            nbors_to_filter = torch.cat((r.unsqueeze(1), nbors.unsqueeze(1), t), dim=-1)

        batch_size = len(entities)

        # extract neighbours of the form (r,e,t) from dict,
        # get a list of tensors and stack(concat) them. Size after concat is say N.
        nbor_list = []
        for idx, i in enumerate(entities):
            nbors_of_i = nbor_dict[i.item()]
            if filter is not None:
                # --filter out the query from neighbour set-- #
                x = (nbors_of_i[:, :] != nbors_to_filter[idx])
                x = (x != 0).sum(dim=1).nonzero().squeeze()
                nbors_of_i = nbors_of_i[x]
                if len(nbors_of_i.shape) == 1:
                    nbors_of_i = nbors_of_i.unsqueeze(0)
                # pdb.set_trace()
                # ------------------------------------------ #

            nbor_list.append(nbors_of_i)

    

        entity_nbors = torch.cat(nbor_list, dim=0)

        entity_degrees = degrees[entities]
        if filter is not None:
            entity_degrees -= 1  # positive sample filtered for each entity, hence reduce degree by 1

        indices = np.repeat(np.arange(batch_size), entity_degrees.cpu())
       

        return entity_nbors, torch.tensor(indices)

    def pairwise_scoring_gadget(self, r_query, r_link, time_diff, mode, device):
        time_diff = time_diff.squeeze()
        if (mode == 'subject'):
            mean = self.mean_r_r_sub[r_link, r_query]
            var = self.var_r_r_sub[r_link, r_query]
            offset = self.offset_r_r_sub[r_link, r_query]
            #mask = self.mask_r_r_sub[r_link, r_query]
        else:
            mean = self.mean_r_r_obj[r_link, r_query]
            var = self.var_r_r_obj[r_link, r_query]
            offset = self.offset_r_r_obj[r_link, r_query]
            #mask = self.mask_r_r_obj[r_link, r_query]


        # --compute prob density-- #
        
        x = -(time_diff - mean) ** 2
      

        x = x / (2 * var)
        prob = torch.exp(x)
        # -------------------- #

   
        prob = prob * self.mask_r_r[r_link,r_query].cuda(device=device)

        prob = prob + offset

    

        return prob 
    def scoring_gadget_recurrent(self, r_query, time_diff, mode, device): 
       
        time_diff = time_diff.squeeze()
        
        if (mode=='subject'):
            mean = self.mean_r_sub[r_query]
            var = self.var_r_sub[r_query]
            offset = self.offset_r_sub[r_query]
            weights = self.W_r_sub[r_query]
        else:
            mean = self.mean_r_obj[r_query]
            var = self.var_r_obj[r_query]
            offset = self.offset_r_obj[r_query]
            weights = self.W_r_obj[r_query]
       
        x = -(time_diff - mean) ** 2
      

        x = x / (2 * var)
        prob = torch.exp(x)
        prob *= weights
        # -------------------- #

        prob = prob + offset

        return prob
        # '''
    def compute_scores_recurrent(self,s,r,o,t,mode,device,eval, predict_time): 
        s = s.squeeze()
        r = r.squeeze()
        o = o.squeeze()
        t = t.squeeze()

        if mode == 'subject':
            # --compute scores for neighbourhood of s--#
            entities = s
            nbors = o

        elif mode == 'object':
            # --compute scores for neighbourhood of o--#
            entities = o
            nbors = s

        else:
            raise Exception('Unknown mode')


        num_samples = len(entities)

        if not eval:
            filter = None
          
            entity_nbors, indices = self.get_nbors_indices(entities, mode=mode, filter=filter)
         
        else:
            entity_nbors, indices = self.eval_tensors[mode]  # use pre-constructed tensors
            indices = indices.cuda(device=device)
            batch_size = num_samples / self.num_entities
            all_nbors = len(entity_nbors) / self.eval_batch_size
            
            entity_nbors = entity_nbors[:int(all_nbors * batch_size)]
            indices = indices[:int(all_nbors * batch_size)]

    
        t_repeated = t[indices]  # torch.index_select(t, 0, indices)
        r_repeated = r[indices]  # torch.index_select(r, 0, indices)
        e_repeated = nbors[indices]  # torch.index_select(r, 0, indices)

      

        time_idx = time_index["t_i"]  # pick start year from time interval id
        r_time = self.t1_emb.index_select(0,entity_nbors[:, 2 + time_idx].cuda(device=device))
        
        if not predict_time:
            query_time = self.t2_emb(t_repeated)
        else:
            query_time = t_repeated.unsqueeze(1)

        # pdb.set_trace()
        time_diff = r_time - query_time

  
        nbor_index = {"r": 0, "e": 1, "t": 2}
        # print("Computing scores")
        r_query = r_repeated
        r_link = entity_nbors[:, nbor_index["r"]]
        e_query = e_repeated
        e_link = entity_nbors[:, nbor_index["e"]]

        time_diff = time_diff.float()  # .unsqueeze_(-1)

        # 1 if same r,e seen at different time (time_diff is non-zero)

        # if self.gadget_type == 'recurring-fact':
        repeated_fact = (r_link.cuda(device=device) == r_query.cuda(device=device)) & (e_link.cuda(device=device) == e_query.cuda(device=device)) & (time_diff != 0).squeeze()
        # elif self.gadget_type == 'recurring-relation':
        #     repeated_fact = (r_link == r_query) & (time_diff != 0).squeeze()

        repeated_fact = repeated_fact.float()
        indices       = indices.cuda(device=device)
        # pdb.set_trace()

        # compress query_fact into binary tensor of length num_samples,
        # with 1 for the sample which has at least one repeated fact.
        # use index_add for this. call it eligible_samples
        eligible_samples = torch.zeros(num_samples).cuda(device=device)
       
        eligible_samples = eligible_samples.index_add(0, indices, repeated_fact)
        eligible_samples[eligible_samples != 0] = 1

        # find smallest absolute time_diff for each sample (closest repeated fact)
        # call it smallest_time_diff
        smallest_time_diff = np.ones(num_samples)*1e5
        time_diff = torch.abs(time_diff).squeeze()

        non_repeated = (repeated_fact == 0).nonzero().squeeze()
        time_diff.scatter_(0, non_repeated, 1e5) # so that non-eligible facts don't interfere with the min operation

        time_diff_np = time_diff.cpu().numpy() #if self.load_to_gpu else time_diff.numpy()
        indices_np = indices.cpu().numpy() #if self.load_to_gpu else indices.numpy()
        np.minimum.at(smallest_time_diff, indices_np, time_diff_np) # pytorch doesn't have a function for this yet

        smallest_time_diff = torch.tensor(smallest_time_diff).float().cuda(device=device) # this takes time if self.load_to_gpu is True! 
                                                                                                          # find an alternative!
        scores = self.scoring_gadget_recurrent(r, smallest_time_diff, mode, device)
        

        # multiply scores with eligible_sample, to give zero scores
        # for non-eligible samples (i.e. for which entities have not been seen with same (r,e)
        final_scores = scores * eligible_samples
        
        #final_scores_temp = torch.zeros(final_scores.shape[0]+2).cuda(device=device)
        #final_scores_temp[0:scores.shape[0]] = final_scores[0:final_scores.shape[0]]
        #final_scores = final_scores_temp
        # ---------------------------------------#

        return final_scores

    def compute_scores_pairs(self,s,r,o,t,mode,device,eval,predict_time):
        s = s.squeeze()
        r = r.squeeze()
        o = o.squeeze()
        t = t.squeeze()

        if mode == 'subject':
            # --compute scores for neighbourhood of s--#
            entities = s
            nbors = o
            #U2_scoring_gadget = self.U2_scoring_gadget[mode]
            #pairwise_scoring_gadget = self.pairwise_scoring_gadget[mode]
            weights = self.W_sub

        elif mode == 'object':
            # --compute scores for neighbourhood of o--#
            entities = o
            nbors = s
            #U2_scoring_gadget = self.U2_scoring_gadget[mode]
            #pairwise_scoring_gadget = self.pairwise_scoring_gadget[mode]
            weights = self.W_obj

        else:
            raise Exception('Unknown mode')

        num_samples = len(entities)

        if not eval:
            filter = None
            #if positive_samples:
             #   filter = (r, nbors, t)
                # pass
            entity_nbors, indices = self.get_nbors_indices(entities, mode=mode, filter=filter) # code this
            indices = indices.cuda(device=device)
            # if positive_samples:
            #     pdb.set_trace()
        else:
            entity_nbors, indices = self.eval_tensors[mode]  # use pre-constructed tensors
            indices = indices.cuda(device=device)
            batch_size = num_samples / self.num_entities
            all_nbors = len(entity_nbors) / self.eval_batch_size

            entity_nbors = entity_nbors[:int(all_nbors * batch_size)]
            indices = indices[:int(all_nbors * batch_size)]

        # use indices to repeat t and r appropriate number of times
        t_repeated = t[indices]  # torch.index_select(t, 0, indices)
        r_repeated = r[indices]  # torch.index_select(r, 0, indices)
        e_repeated = nbors[indices]  # torch.index_select(r, 0, indices)

        # print("t after repeating:{}, r after repeating:{}".format(t_repeated.shape, r_repeated.shape))

        # compute time diff
        # time_idx = time_index["t_s"]  # pick exact year instead of bin?
        # r_time = entity_nbors[:, 2 + time_idx]
        # query_time = t_repeated[:, time_idx]
        time_idx = time_index["t_i"]  # pick start year from time interval id
        r_time = self.t1_emb.index_select(0,entity_nbors[:, 2 + time_idx].cuda(device=device))

        if not predict_time:
            query_time = self.t2_emb(t_repeated)
        else:
            query_time = t_repeated.unsqueeze(1)
        # pdb.set_trace()
        # print("r_time shape:{}, query_time:{}".format(r_time.shape, query_time.shape))
        # print(query_time)
        # print(time_diff)

        # we have all features! Now compute scores (N scores)
        # features are- r_query, r_link, diff, entity
        nbor_index = {"r": 0, "e": 1, "t": 2}
        # print("Computing scores")
        r_query = r_repeated
        r_link = entity_nbors[:, nbor_index["r"]]
        e_query = e_repeated
        e_link = entity_nbors[:, nbor_index["e"]]

        #pairwise_scoring_gadget = self.scoring_gadget[mode]
        time_diff = r_time - query_time
        time_diff = time_diff.float()  # .unsqueeze_(-1)
        # code pairwise scoring gadget
        pairwise_scores = self.pairwise_scoring_gadget(r_query, r_link, time_diff, mode, device)

        # we have scores, now compute soft attention weights for them.
        wt = weights[r_link, r_query]

        # '''
        # compute softmax
        wt = wt - torch.max(wt)
        wt = torch.exp(wt).cuda(device=device)

        wt_sum = (torch.zeros(num_samples)).cuda(device=device)
        wt_sum = wt_sum.index_add(0, indices, wt)
        wt_sum = wt_sum[indices]
        wt = wt / (wt_sum.squeeze())
        # print("Computed weights with softmax")
        # '''
        

        # weights computed, now multiply with scores output from scoring_gadget and compute summation
        final_pairwise_scores = torch.zeros(num_samples).cuda(device=device)
        final_pairwise_scores = final_pairwise_scores.index_add(0, indices, wt * pairwise_scores.squeeze())


        final_scores = final_pairwise_scores

        return final_scores
    def compute_scores(self,s, r, o, t, type_arg, device,mode='subject', eval=True, predict_time='False'):
        if (type_arg == 'pairs'):
            return self.compute_scores_pairs(s,r,o,t,mode,device,eval, predict_time)
        else:
            return self.compute_scores_recurrent(s,r,o,t,mode,device,eval, predict_time)

    def common_forward(self,s,r,o,t,type_arg,device):

        if s is None:  # scores over all entities
            batch_size = len(o)

            # s = func_load_to_gpu((torch.arange(self.entity_count)).repeat(batch_size), self.load_to_gpu)
            # s = (torch.arange(self.entity_count)).repeat(batch_size)
            s = ((torch.arange(self.num_entities)).repeat(batch_size)).cuda(device=device)
            #s = self.eval_ids[:batch_size * self.entity_count]

            #o = func_load_to_gpu(torch.from_numpy(o.cpu().numpy().repeat(self.entity_count)), self.load_to_gpu)
            o = (torch.from_numpy(o.cpu().numpy().repeat(self.num_entities))).cuda(device=device)
            r = (torch.from_numpy(r.cpu().numpy().repeat(self.num_entities))).cuda(device=device)
            t = (torch.from_numpy(t.cpu().numpy().repeat(self.num_entities, axis=0))).cuda(device=device)

            # print("s shape:{}, r shape:{}, o shape:{}, t shape:{}".format(s.shape, r.shape, o.shape, t.shape))

            sub_scores = self.compute_scores(s, r, o, t, type_arg, device, mode='subject', eval=True, predict_time='False')

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, type_arg, device, mode='object', eval=True, predict_time='False')
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape((batch_size, self.num_entities))
            scores_temp = scores

        elif o is None:  # scores over all entities
            batch_size = len(s)
            #print(batch_size)
            # o = func_load_to_gpu((torch.arange(self.entity_count)).repeat(batch_size), self.load_to_gpu)
            # o = (torch.arange(self.entity_count)).repeat(batch_size)
            o = ((torch.arange(self.num_entities)).repeat(batch_size)).cuda(device=device)
            #s = self.eval_ids[:batch_size * self.entity_count]

            s = (torch.from_numpy(s.cpu().numpy().repeat(self.num_entities))).cuda(device=device)
            r = (torch.from_numpy(r.cpu().numpy().repeat(self.num_entities))).cuda(device=device)
            t = (torch.from_numpy(t.cpu().numpy().repeat(self.num_entities, axis=0))).cuda(device=device)
            #print(s.shape,r.shape,o.shape,t.shape)
            sub_scores = self.compute_scores(s, r, o, t, type_arg,device, mode='subject', eval=True)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, type_arg,device, mode='object', eval=True)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores
            #print(scores.shape)
            #print(scores)

            scores = scores.reshape((batch_size, self.num_entities))
            scores_temp = scores
        
        elif t is None:
            batch_size = len(s)
            
            #year2id_mat = self.convert_dict2mat(self.dateYear2id)
            self.times = (torch.from_numpy(self.convert_dict2mat(self.dateYear2id))).cuda(device=device)
            num_times = len(self.times)

            
            s = s.repeat(1, num_times).flatten()
            r = r.repeat(1, num_times).flatten()
            o = o.repeat(1, num_times).flatten()

            # pdb.set_trace()
            t = (self.times).float()
            t = t.repeat(batch_size)
            
            sub_scores = self.compute_scores(s, r, o, t, type_arg, device, mode='subject', eval=False)

            if self.use_obj_scores:
                obj_scores = self.compute_scores(s, r, o, t, type_arg,device, mode='object', eval=False)
                scores = sub_scores + obj_scores
            else:
                scores = sub_scores

            scores = scores.reshape(batch_size, num_times)  
            scores_temp = torch.zeros(size = (batch_size, scores.shape[1]+2)).cuda(device=device)
            scores_temp[:, 0:scores.shape[1]] = scores[:, 0:scores.shape[1]]
        
        return scores_temp

    @torch.no_grad()
    def forward(self, all_h, all_r, all_t, all_t1, all_t2, predict_time,extra_param):

        all_scores = []
        
        if not predict_time:
            for i in range(len(all_h)):
                num_times = all_t2[i] - all_t1[i] + 1
                t_i = (torch.arange(all_t1[i],all_t2[i]+1,device = all_h.device)).unsqueeze(1) # Interval_length x 1
                r_i = all_r[i].repeat(num_times,1)
                h_i = all_h[i].repeat(num_times,1)
                t_start_rep = all_t1[i].repeat(num_times,1)
                pairs_scores = 0.0
                recurrent_scores = 0.0
               
                if (all_r[0] >=self.relation_count-1): #
                    if (self.use_inverse):
                        scores = self.scoring_function(h_i,r_i,None,t_i,predict_time)
                    else:
                        scores = self.scoring_function(None,r_i - (self.relation_count-1),h_i,t_i,predict_time)
                    if (self.recurrent_wt!=0):
                        recurrent_scores = (self.common_forward(None,all_r[i].reshape(1,1)-(self.relation_count-1) ,all_h[i].reshape(1,1),all_t1[i].reshape(1,1),'recurrent',device=all_h.device)).repeat(num_times,1) if self.recurrent_wt else 0.0
                    if (self.pairs_wt!=0):
                        pairs_scores = (self.common_forward(None,all_r[i].reshape(1,1)- (self.relation_count-1),all_h[i].reshape(1,1),all_t1[i].reshape(1,1),'pairs',device=all_h.device)).repeat(num_times,1) if self.pairs_wt else 0.0
                else:
                    
                    scores = self.scoring_function(h_i,r_i,None,t_i,predict_time)
                    if (self.recurrent_wt!=0):
                        recurrent_scores = (self.common_forward(all_h[i].reshape(1,1),all_r[i].reshape(1,1),None,all_t1[i].reshape(1,1),'recurrent',device=all_h.device)).repeat(num_times,1) if self.recurrent_wt else 0.0
                    if (self.pairs_wt!=0):
                        pairs_scores = (self.common_forward(all_h[i].reshape(1,1),all_r[i].reshape(1,1),None,all_t1[i].reshape(1,1),'pairs',device=all_h.device)).repeat(num_times,1) if self.pairs_wt else 0.0
                

            
                all_scores.append((scores + self.recurrent_wt*recurrent_scores +self.pairs_wt*pairs_scores).sum(dim=0))
                    #all_scores.append((scores.sum(dim=0)))

                del scores, recurrent_scores, pairs_scores
        else:
            #print("all_h: {} ,all_r: {} ,all_t: {} all_h.shape:{}, all_r.shape: {} all_t.shape: {}".format(all_h,all_r,all_t, all_h.shape, all_r.shape, all_t.shape)) #all_h: tensor([9150], device='cuda:0') ,all_r: tensor([0], device='cuda:0') ,all_t: tensor([9150], device='cuda:0') all_h.shape:torch.Size([1]), all_r.shape: torch.Size([1]) all_t.shape: torch.Size([1])

            for i in range(len(all_h)):
                h_i = all_h[i]
                r_i = all_r[i]                
                t_i = all_t[i]
                pairs_scores = 0.0
                recurrent_scores = 0.0
                if (self.recurrent_wt!=0):
                    recurrent_scores = (self.common_forward(all_h[i].reshape(1,1),all_r[i].reshape(1,1),all_t[i].reshape(1,1),None,'recurrent',device=all_h.device)) if self.recurrent_wt else 0.0
                if (self.pairs_wt!=0):
                    pairs_scores     = (self.common_forward(all_h[i].reshape(1,1),all_r[i].reshape(1,1),all_t[i].reshape(1,1),None,'pairs',    device=all_h.device)) if self.pairs_wt     else 0.0
             
            

                
                all_scores.append((self.scoring_function(h_i,r_i,t_i,None,predict_time) + self.recurrent_wt*recurrent_scores +self.pairs_wt*pairs_scores).sum(dim=0))

        all_scores = torch.stack(all_scores)
       
        return all_scores
       
