from dicee import QueryGenerator
import pickle


q = QueryGenerator(dataset="UMLS", seed=42, gen_train=False, gen_valid=False, gen_test=True)
# Either generate queries and save it at the given path
q.save_queries(query_type="2in",gen_num=10,save_path= "/Users/sourabh/Desktop/dice-embeddings/dice-embeddings/KGs/WN18")
# or else get it as dicts to answer queries directly using pre_trained_KGE.answer_multi_hop_query(....)
query_dict, easy_answers, false_positives , hard_answers=q.get_queries(query_type="2in",gen_num=10)

print(query_dict,easy_answers,false_positives,hard_answers)
