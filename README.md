# NeuSTIP: A Neuro-Symbolic Model for Link and Time Prediction in Temporal Knowledge Graphs

## Setup and running the model

Create workspace directory inside codes-parameter:

mkdir workspace

Create Rules directory inside respective data folders:

(cd data/YAGO11k) or (cd data/WIKIDATA12k)

mkdir Rules_3_temp

Variables: (present inside {.} )

dataset_name from {YAGO11k, WIKIDATA12k}
num_relations : 20 for YAGO11k, 48 for WIKIDATA12k
eta: weight of KGE model
char_string_link : {dataset_name}_{eta}_0.001_5000_temp_3
char_string_time: {dataset_name}_{eta}_0.001_2000_temp_use_duration_3_all


Structure Learning: 


cd codes_structure
python all_walks.py {dataset_name} {rule_file} 1 3 0 0 0 
python rules_for_model.py {rule_file} Rules_3_temp {dataset_name} {num_relations}

Parameter learning:

Link Prediction:

python run.py {dataset_name} {eta} 0 {num_relations} 0 link 0 3 0 1 0

Time Prediction:

python run.py {dataset_name} {eta} 0 {num_relations/2} 0 time 1 3 0 1 0

To output final results:

Link prediction

python link_metrics.py {char_string_link} {num_relations}

Time Prediction

python time_metrics.py {char_string_time} {num_relations/2} {dataset_name}
