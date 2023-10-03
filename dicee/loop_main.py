import subprocess
import sys
from static_funcs_training import evaluate_lp
import json
import pandas as pd

parameter_values = [0,1]

results = dict()

evaluate_lp_outputs = []
df = pd.DataFrame()

for p in parameter_values:
    for q in parameter_values:
        for r in parameter_values:
           
           if 16%(p+q+r+1) == 0:
                subprocess_output = subprocess.run(["python", "/home/dice/Desktop/dice-embeddings/main.py" ,"--p",str(p)\
                                ,"--q",str(q), "--r", str(r), "--neg_ratio",str(50),"--batch_size",str(32),"--num_epochs",str(100)],stdout=subprocess.PIPE,universal_newlines=True,)
                eval_output = subprocess_output.stdout.strip()

                eval_output = eval_output[-147:-30]

                #print(eval_output)

                start_index = eval_output.find("{")

                data_str = eval_output[start_index:]

                data_str = data_str.replace("'", "\"")

                data_dict = json.loads(data_str)

                results.update({'p':p,'q':q,'r':r})
                results.update(data_dict)
                
                dfi = pd.DataFrame(results, index = [0])

                df = pd.concat([df,dfi])

print(df)          


#print(evaluate_lp_outputs)
    
    
