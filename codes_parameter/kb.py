import numpy
import torch
YEARMIN= 0
YEARMAX= 3000
class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """

    def __init__(self, datamap, filename, add_unknowns: bool = True,
                 nonoov_entity_count: int = None,
                 use_time_tokenizer: bool = False) -> object:
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entities are to be acknowledged or put as <UNK> token.
        """

        self.datamap = datamap
        self.use_time_tokenizer = use_time_tokenizer
        self.filename=filename

        facts_time_tokens = []  # for TA-x models
        facts = []

        if filename is None:
            return

        # --for time--#
        self.unk_time_str = 'UNK-TIME'  # for facts with no time stamp or invalid time stamps
        # ----------- #

        self.nonoov_entity_count = 0 if nonoov_entity_count is None else nonoov_entity_count

        print("KB", filename, add_unknowns)

        with open(filename) as f:
            lines = f.readlines()
            lines = [l.strip("\n").split("\t") for l in lines]
            cnt = 0
            # ----------- #
            count_missed_facts = 0
            for l in lines:  # preparing data

                # Main Job
                time_str = 'UNK-TIME'
                if len(l)==3: # for non-temporal datasets
                    facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                    self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                    self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                    0, 0, 0, 0, 0,0])
                else:
                    if self.datamap.use_time_interval:
                        if len(l) == 5:  # timestamp of the form "occursSince <YEAR>" or "<YEAR1> <YEAR2>"
                            print("Inside use_time_interval")

                            
                            t_start_lbl, t_end_lbl = self.get_span_ids(l[3], l[4])
                            if t_start_lbl == "" or t_end_lbl == "":
                                count_missed_facts += 1
                                continue
                            assert t_end_lbl >= t_start_lbl
                            start, end = self.get_date_range(l)
                            time_interval_str_id = self.datamap.dateYears2intervalId.get((start, end), len(
                                self.datamap.dateYears2intervalId) - 1)  # self.dateYears2intervalId[(start,end)
                            # ipdb.set_trace()

                            time_str = '\t'.join(l[3:])
                            time_str_id = self.datamap.timeStr2Id[time_str]

                            facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                        self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                        self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                        t_start_lbl, t_start_lbl, t_end_lbl, t_end_lbl, time_str_id,
                                        time_interval_str_id])
                        elif len(l) != 3:
                            print("Unknown time format")
                            raise Exception
                        else:
                            count_missed_facts += 1
                    else:
                        if len(l) > 3:
                            #print("Inside len(l) > 3")
                            
                            #start, end = self.get_date_range(l)
                            start = int(l[3])
                            end = int(l[4])
                            start_id, end_id = (self.datamap.dateYear2id[start], self.datamap.dateYear2id[end])
                            time_interval_str_id = self.datamap.dateYears2intervalId.get((start, end), len(
                                self.datamap.dateYears2intervalId) - 1)  # self.dateYears2intervalId[(start,end)]

                            time_str = '\t'.join(l[3:])  # l[-1]
                            time_str_id = self.datamap.timeStr2Id[time_str]

                            facts.append([self.datamap.entity_map.get(l[0], len(self.datamap.entity_map) - 1),
                                        self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1),
                                        self.datamap.entity_map.get(l[2], len(self.datamap.entity_map) - 1),
                                        start_id, start_id, end_id, end_id, time_str_id, time_interval_str_id])
                            if (self.datamap.relation_map.get(l[1], len(self.datamap.relation_map) - 1) == 0):
                                cnt = cnt + 1
                          
                        elif len(l) < 3:
                            print("Bad data: Unknown time format")
                            raise Exception

                if self.use_time_tokenizer:
                    # time_tokens=tokenize_time(time,filename)
                    time_tokens = tokenize_time(time_str, filename)
                    facts_time_tokens.append(time_tokens)

        self.facts_time_tokens = numpy.array(facts_time_tokens, dtype='int64')
        self.facts = numpy.array(facts, dtype='int64')
        print("Data Size:", filename, self.facts.shape)

    def get_date_range(self, fact):
        if len(fact) == 3:
            t1 = t2 = "###"
        elif len(fact) == 4:
            _, _, _, t1 = fact
            t2 = t1
        else:
            _, _, _, t1, t2 = fact
        # start = self.check_date_validity(t1)
        # end = self.check_date_validity(t2)
        start = self.datamap.check_date_validity(t1)
        end = self.datamap.check_date_validity(t2)

        if t1 == '0':
            start =0
        if (start != -1 and end != -1) and (start > end):
            end = start
        if start == -1 and end != -1:
            start = YEARMIN  # self.unk_time_str#end
        elif start != -1 and end == -1:
            end = YEARMAX  # self.unk_time_str#start
        elif start == -1 and end == -1:
            # start = end = self.unk_time_str
            start = YEARMIN
            # end = YEARMAX
            end = YEARMIN

        return start, end