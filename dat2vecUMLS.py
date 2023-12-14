from keci_r_MRR import Keci_exp
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import os
import shutil
import pandas as pd
import pickle

#######################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--kg", type=str, default='')
parser.add_argument("--num_subgraphs", type=int, default=5, help = "Number of subfolder for the given KG to be created")
parser.add_argument("--step", type = str, default='training', help = 'Do you want to predict or train', choices = ["training","prediction"] )
parser.add_argument("--tensor_size", type = int, default = 5000, help = 'Tensor data size as input in the NN')

args = parser.parse_args()


torch_seed = 0
torch.manual_seed(torch_seed)
python_random_seed = 0
random.seed(python_random_seed)

########################################################################################################################


def create_data(kg,num_subgraphs,tensor_size,step):
    '''This function takes the list of subfolder name '''
    
    N = tensor_size
    original_kg_folder = f"../dice-embeddings/KGs/{kg}"
    output_base_folder = f"../dice-embeddings/d2v_Experiments/{kg}_subgraph"
    num_subgraphs =num_subgraphs

    l = random_walk_subgraph(original_kg_folder, output_base_folder, num_subgraphs)
    print(f"Folders for {kg} created and files copied.")
    
    
    path_main = "../dice-embeddings/main.py"
    data_dict = {}
    for sub_kg in l:
        
        folder_name = f"../dice-embeddings/d2v_Experiments/Experiments_{sub_kg}_local"
        Experiments_path=f"../dice-embeddings/d2v_Experiments/Experiments_{sub_kg}_local"
        path_dataset =  f"../dice-embeddings/d2v_Experiments/{sub_kg}"
        Num_epochs = 250
        Batch_size = 1024

        if step == "training":
            
            dat = Keci_exp(emb_dim=16,path_main = path_main,folder_name = folder_name,Experiments_path=Experiments_path,\
                       num_epochs=250,batch_size=1024, scoring_technique= "KvsAll",path_dataset=path_dataset)

            (p,q,r), _ = dat.exaustive_search_local(params_range = range(2)) # take the result of the exhaustive search in [0 4].

            file_path = os.path.join(Experiments_path, f"{1}_{1}_{1}") 

            
            D_i = tensor_data(file_path,N)
            
            data_dict.update({D_i:(p,q,r)})
                
        if step == "prediction":

            (p,q,r) = (1, 1, 1)

            file_path = os.path.join(Experiments_path, f"{p}_{q}_{r}")  

            D_i = tensor_data(file_path,N)


            data_dict.update(D_i)
            
    print(data_dict)
    file_name = f"{kg}_data.pth" 
    torch.save(data_dict, file_name) #save as pytorch

    return data_dict


def tensor_data(file_path,N):
    
    for folder in os.listdir(file_path):
        


        folder_path  = os.path.join(file_path, folder)
        train_path   = os.path.join(folder_path, 'train_set.npy')
        path_ent_idx = os.path.join(folder_path, 'entity_to_idx.p')
        path_rel_idx = os.path.join(folder_path, 'relation_to_idx.p')
        path_ent_emb = os.path.join(folder_path, 'Keci_r_entity_embeddings.csv')
        path_rel_emb = os.path.join(folder_path, 'Keci_r_relation_embeddings.csv')

        train_data = np.load(train_path)


        D_i = torch.zeros(N,48) #N is the size of the tensor
        for i in range(len(D_i)):
            indx_triple = train_data[i,:]
            h,r,t = ent_rel_emb(indx_triple,path_ent_idx,path_rel_idx,path_ent_emb,path_rel_emb)
            h_r_t = torch.concatenate((h,r,t),dim=0)
            D_i[i,:] = h_r_t
            
    return D_i


def ent_rel_emb(indx_triple,path_ent_idx,path_rel_idx,path_ent_emb,path_rel_emb):

    # Here we retrieve final entities embedings we just need the indx (value == indx) of and entity from the file above:
    
    with open(path_ent_idx, 'rb') as file:
        ent_to_idx = pickle.load(file)

    head = next((key for key, value in ent_to_idx.items() if value == indx_triple[0]), None)
    tail = next((key for key, value in ent_to_idx.items() if value == indx_triple[2]), None)
    
    head_emb = pd.read_csv(path_ent_emb, index_col=0)
    head_id_to_extract = head
    vector_head_emb = head_emb.loc[head_id_to_extract].values

    tail_emb = pd.read_csv(path_ent_emb, index_col=0)
    tail_id_to_extract = tail
    vector_tail_emb = tail_emb.loc[tail_id_to_extract].values




    # Here we retrieve final entities embedings we just need the indx of a relation:
    
    with open(path_rel_idx, 'rb') as file:
        rel_to_idx = pickle.load(file)

    rel = next((key for key, value in rel_to_idx.items() if value == indx_triple[1]), None)
    rel_emb = pd.read_csv(path_rel_emb, index_col=0)


    rel_id_to_extract = rel
    vector_rel_emb = rel_emb.loc[rel_id_to_extract].values

    return  torch.tensor(vector_head_emb), torch.tensor(vector_rel_emb), torch.tensor(vector_tail_emb)


def random_walk_subgraph(original_folder, output_base_folder, num_subgraphs):
    # returns the name of folders where the subgraphes has been created.
    l = []
    for i in range(1, num_subgraphs + 1):
        output_folder = f"{output_base_folder}_{i}"
        name = os.path.basename(output_folder)
        l.append(name)
        os.makedirs(output_folder, exist_ok=True)

        # Copy files to the subgraph folder
        for filename in ["test.txt", "train.txt", "valid.txt"]:
            shutil.copy(os.path.join(original_folder, filename), os.path.join(output_folder, filename))
            
            # Perform random walk and sample subgraph for every files
            sampled_subgraph = random_walk(original_folder, filename)

            # Write subgraph to text.txt
            with open(os.path.join(output_folder, filename), "w") as text_file:
                for triple in sampled_subgraph:
                    text_file.write("\t".join(map(str, triple)) + "\n")
                    
    return l

def random_walk(original_folder,filename): # This is implemented by following the Algorithm in https://dl.acm.org/doi/pdf/10.1145/3583780.3615158
    # Load KG from text.txt
    KG = set()
    
    with open(os.path.join(original_folder, filename), "r") as file:
        for line in file:
            triple = tuple(map(str, line.strip().split("\t")))
            KG.add(triple)

    if filename == 'train.txt':
        
        ratio = 5116/len(KG)
        
    elif filename == 'test.txt':
        
        ratio = 661/len(KG)
    else:
        
        ratio = 652/len(KG)
        

    # Perform random walk
    E = set()
    kg = []#set()    #new empty kg 
    start_entity = random.choice([triple[0] for triple in KG])
   
    E.add(start_entity)
    
    iterations = 0
    max_iterations = 1000
    
    
    while  len(kg) < ratio * len(KG): #and iterations < max_iterations:
     
        S = set([(h,r,t) for (h,r,t) in KG if h ==start_entity])

        if not S:
            start_entity = random.choice([triple[0] for triple in KG])
            E.add(start_entity)
        else:
           
            (h_,r_,t_) = random.choice(list(S))
            
            E.add(t_)
         
            kg.extend(S)
            start_entity = t_
    
        
        iterations += 1
    return kg

    
    
data =  create_data(f'{args.kg}',args.num_subgraphs,args.tensor_size,args.step)

    

# Original folder path
#original_folder_path = f'../dice-embeddings/KGs/{args.kg}' 

# # List of file names
# file_names = ['test.txt', 'train.txt', 'valid.txt']

# # Name to consider as arguments: 6, UMLS

# # Create 5 target folders
# l = []
# Num_folder = 5
# for i in range(1, Num_folder+1):
#     target_folder_path = f'{args.kg}_data2vec/{args.kg}_{i}' # The experiments are saved iside this folder
#     os.makedirs(target_folder_path, exist_ok=True)
    
#     l.append(f'{args.kg}_{i}')

#     # Copy 1/5 of the elements to each target folder
#     for file_name in file_names:
#         original_file_path = os.path.join(original_folder_path, file_name)
#         target_file_path = os.path.join(target_folder_path, file_name)

#         with open(original_file_path, 'r') as original_file:
#             lines = original_file.readlines()
#             # Calculate the number of lines to copy (1/5 of the total)
#             num_lines_to_copy = len(lines) // (Num_folder)

#             # Randomly choose lines to copy
#             random_lines = random.sample(lines, num_lines_to_copy)

#         with open(target_file_path, 'w') as target_file:
#             target_file.writelines(random_lines)

# #print(l)