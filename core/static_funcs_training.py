def efficient_zero_grad(model):
    # Use this instead of
    # self.optimizer.zero_grad()
    #
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
    for param in model.parameters():
        param.grad = None
