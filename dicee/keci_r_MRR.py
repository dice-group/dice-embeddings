import subprocess
import json

def MRR(p,q,r):

    subprocess_output = subprocess.run(["python", "/home/dice/Desktop/dice-embeddings/main.py" ,"--p",str(p)\
                            ,"--q",str(q), "--r", str(r), "--neg_ratio",str(50),"--batch_size",str(32),"--num_epochs",str(100)],stdout=subprocess.PIPE,universal_newlines=True,)
    eval_output = subprocess_output.stdout.strip()
    
    result_output = eval_output[-147:-30]

    start_index = result_output.find("{")

    data_str = result_output[start_index:]

    data_str = data_str.replace("'", "\"")

    data_dict = json.loads(data_str)

    return data_dict['MRR']

