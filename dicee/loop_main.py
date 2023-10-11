from keci_r_MRR import Keci_exp


dat = Keci_exp(emb_dim=2,path_main="/home/dice/Desktop/dice-embeddings/main.py",num_epochs=1, batch_size=32, neg_ratio=50)

print(dat.results_keci())
print(dat.params_search((1,1,0)))






        

        

