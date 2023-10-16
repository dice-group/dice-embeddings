from keci_r_MRR import Keci_exp


dat = Keci_exp(emb_dim=4,path_main="/home/dice/Desktop/dice-embeddings/main.py",folder_name="Experiments_UMLS",Experiments_path="/home/dice/Desktop/dice-embeddings/Experiments_UMLS")

print(dat.results_keci(parameter_values=[0,1]))
print(dat.params_search((1,1,0)))






        

        

