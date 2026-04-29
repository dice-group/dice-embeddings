from dicee.models import Keci
import numpy as np
import torch
# Indexed Triples
X=torch.from_numpy(np.array([[0,0,1],
                               [0,1,1],
                               [1,1,1]])).long()

# Labels
y=torch.from_numpy(np.array([1,1,0])).float()

# Model
model=Keci(args={"num_entities":2,"num_relations":2,"embedding_dim":8,"optim":"Adopt"})
# Optim
optim=torch.optim.Adam(model.parameters(),lr=0.1)
#
model.train()
for i in range(10):
    optim.zero_grad()

    yhat=model(X)

    loss=torch.nn.functional.binary_cross_entropy_with_logits(yhat,y)

    print(loss.item())

    loss.backward()

    optim.step()

model.eval()

with torch.no_grad():
    print(model(X).mean())