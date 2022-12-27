# Barlow Twins: Self-Supervised Learning via Redundancy Reduction
class BarlowTwins:
    def __init__(self, model):
        self.name = f'Barlow_{model.name}'
        self.model = model
