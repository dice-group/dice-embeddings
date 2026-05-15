import torch
from pfn.inference import infer
from pfn.model import TriplePFN

ckpt=torch.load('model.pt',map_location='cuda' if torch.cuda.is_available() else 'cpu')
if isinstance(ckpt,dict) and 'hparams' in ckpt:
    m=TriplePFN(**ckpt['hparams'])
    m.load_state_dict(ckpt['state_dict'])
else:
    m=TriplePFN(); m.load_state_dict(ckpt)
m.eval()

support=[]
with open('KGs/Countries-S1/train.txt') as f:
    for line in f:
        p=line.strip().split()
        if len(p)==3:
            support.append(tuple(p))
support=support[:128]

res=infer(m,'slovakia','neighbor',support,k=min(20,len({x for t in support for x in (t[0],t[2])})),device=torch.device('cpu'))
print('top 10 raw logits:')
for e,s in res[:10]:
    print(e, s)