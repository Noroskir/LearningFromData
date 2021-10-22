import numpy as np


class Config:
    def __init__(self, filename):
        self.filename = filename
        self.args = {'eta': float, 'beta1': float,
                     'beta2': float, 'lambda': float,
                     'epochs': int, 'batchsize': int}

    def parse_config_args(self):
        vals = dict()
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                args = l.replace('=', '').split()
                if l[0] == '#':
                    continue
                elif args == []:
                    continue
                if args[0] in self.args:
                    vals[args[0]] = self.args[args[0]](args[1])
                # except Exception:
                #     pass
        return vals
