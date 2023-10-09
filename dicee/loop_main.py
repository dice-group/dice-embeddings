import subprocess
import json
import pandas as pd
from keci_r_MRR import MRR

parameter_values = [0,1]

results = dict()

df = pd.DataFrame()
i = 0 #for indexing
for p in parameter_values:
    for q in parameter_values:
        for r in parameter_values:
           
           if 16%(p+q+r+1) == 0:
                subprocess_output = subprocess.run(["python", "/home/dice/Desktop/dice-embeddings/main.py" ,"--p",str(p)\
                                ,"--q",str(q), "--r", str(r), "--neg_ratio",str(50),"--batch_size",str(32),"--num_epochs",str(100)],stdout=subprocess.PIPE,universal_newlines=True,)
                eval_output = subprocess_output.stdout.strip()

                result_output = eval_output[-147:-30]
                #print(result_output)

                time_output = eval_output[-29:-1]
                #print(time_output)
                parts = time_output.split()
                run_time = float(parts[-2])

                start_index = result_output.find("{")

                data_str = result_output[start_index:]

                data_str = data_str.replace("'", "\"")

                data_dict = json.loads(data_str)

                results.update({'p':p,'q':q,'r':r})
                results.update(data_dict)
                results.update({'time(s)': run_time})
                
                dfi = pd.DataFrame(results, index = [i])

                df = pd.concat([df,dfi])

                i += 1

print(df)          

'''Gradient based searching parameters'''

(opt_p, opt_q, opt_r) = (1,1,1)
p0, q0, r0 = 1, 1, 1
max_MRR = 0
l = []
d = 1

for p in range(p0, d):
  for q in range(q0, d):
    for r in range(q0, d):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)

for p in range(p0, -1, -1):
  for q in range(q0, -1, -1):
    for r in range(q0, -1, -1):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)

for p in range(p0, d):
  for q in range(q0, -1, -1):
    for r in range(r0, -1, -1):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)

for p in range(p0, -1, -1):
  for q in range(q0, d):
    for r in range(r0, -1, -1):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)


for p in range(p0, -1, -1):
  for q in range(q0, -1, -1):
    for r in range(r0, d):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)


for p in range(p0, d):
  for q in range(q0, d):
    for r in range(r0, -1, -1):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)


for p in range(p0, d):
  for q in range(q0, -1, -1):
    for r in range(r0, d):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)


for p in range(p0, -1, -1):
  for q in range(q0, d):
    for r in range(r0, d):
      if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
        l.append((p, q, r))
        Mrr = MRR(p,q,r)

        if Mrr > max_MRR:
          max_MRR = Mrr
          (opt_p, opt_q, opt_r) = (p,q,r)

print(l)
print(max_MRR)
print((opt_p,opt_q,opt_r))


        

        

