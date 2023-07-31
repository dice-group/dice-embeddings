import sys,os
sys.path.append(os.getcwd())

from dicee.knowledge_graph_embeddings import KGE
from dicee.knowledge_graph import KG

if __name__=="__main__":

  path = "E:\\DICEE\\dice-embeddings\\Experiments\\2023-07-29 22-42-44.871546"
  path_dataset_folder = './KGs/UMLS'
  
  pre_trained_kge = KGE(path=path)
  kg = KG(data_dir=path_dataset_folder,eval_model='test')
  
  pre_trained_kge.lp_evaluate(dataset=kg)