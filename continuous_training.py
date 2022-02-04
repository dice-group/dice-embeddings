from core.executer import Execute
from collections import namedtuple

import dask.dataframe as dd
import os
import json


class CustomArg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        return self.kwargs[name]

    def __repr__(self):
        return f'CustomArg at {hex(id(self))}: ' + str(self.kwargs)

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for k,v in self.kwargs.items():
            yield k,v


def load_configuration(p: str) -> namedtuple:
    assert os.path.isfile(p)
    with open(p, 'r') as r:
        args = json.load(r)
    return CustomArg(**args)


# test loading
def parquet_loading(p):
    return dd.read_parquet(p).compute()


class ContinuousExecute(Execute):
    def __init__(self, folder_path: str):
        assert os.path.exists(folder_path)
        assert os.path.isfile(folder_path + '/idx_train_df.gzip')
        assert os.path.isfile(folder_path + '/configuration.json')
        args=load_configuration(folder_path + '/configuration.json')

        args.num_epochs=10
        super().__init__(args, continuous_training=True)


ContinuousExecute(folder_path="DAIKIRI_Storage/2022-02-04 13:53:13.658691").start()
