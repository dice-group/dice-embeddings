import torch
import dicee
from dicee import DistMult
import polars as pl
import time

print("Reading KG...", end=" ")
start_time=time.time()
data = pl.read_parquet("dbpedia-2022-12-nt.parquet.snappy")#,n_rows=100_000_000)
print(f"took {time.time()-start_time}")
print("Unique entities...", end=" ")
start_time=time.time()
unique_entities=pl.concat((data.get_column('subject'),data.get_column('object'))).unique().rename('entity').to_list()
print(f"took {time.time()-start_time}")

print("Unique relations...", end=" ")
start_time=time.time()
unique_relations=data.unique(subset=["relation"]).select("relation").to_series().to_list()
print(f"took {time.time()-start_time}")

print("Entity index mapping...", end=" ")
start_time=time.time()
entity_to_idx={ ent:idx for idx, ent in enumerate(unique_entities)}

print("Relation index mapping...", end=" ")
start_time=time.time()
rel_to_idx={ rel:idx for idx, rel in enumerate(unique_relations)}
print(f"took {time.time()-start_time}")

print("Construcing training data...", end=" ")
start_time=time.time()
data=data.with_columns( pl.col("subject").map_dict(entity_to_idx).alias("subject"), pl.col("relation").map_dict(rel_to_idx).alias("relation"), pl.col("object").map_dict(entity_to_idx).alias("object")).to_numpy()
print(f"took {time.time()-start_time}")
print("Deleting dataframe...", end=" ")


print("KGE model...", end=" ")
start_time=time.time()
model=DistMult(args={"num_entities":len(entity_to_idx),"num_relations":len(rel_to_idx),"embedding_dim":20})

model.to(torch.device("cuda:0"))

print(f"took {time.time()-start_time}", end=" ")
print("Optimizer...")
start_time=time.time()
optimizer = model.configure_optimizers()


loss_function = model.loss_function

batch_size=10_000
start_index=0
print("Training...")
i_step=0
for x in data[start_index:start_index+batch_size]:
    
    optimizer.zero_grad(set_to_none=True)

    x=torch.from_numpy(x).int().to("cuda:0").unsqueeze(0)
    
    yhat=model.forward(x)
    y=torch.ones(len(yhat),device="cuda:0")
    


    batch_positive_loss=loss_function(yhat,y)
    
    if i_step%100==0:    
        print(i_step,batch_size,batch_positive_loss)

    start_index+=batch_size
    i_step +=1
    
    if i_step<=10:
        batch_size+=batch_size
    
    if i_step==1000:
        break


print(torch.cuda.memory_summary())
print(torch.cuda.mem_get_info())
print('DONE')
