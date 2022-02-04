class CustomArg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def update(self, x: dict):
        self.kwargs.update(x)

    def __getattr__(self, name):
        return self.kwargs[name]

    def __repr__(self):
        return f'CustomArg at {hex(id(self))}: ' + str(self.kwargs)

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for k, v in self.kwargs.items():
            yield k, v
