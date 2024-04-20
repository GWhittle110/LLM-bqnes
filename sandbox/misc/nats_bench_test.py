import os
from nats_bench import create
from dotenv import dotenv_values
from xautodl.models import get_cell_based_tiny_net

os.environ["HOME"] = dotenv_values()["HOME"]

api = create(dotenv_values()["HOME"] + "\\.torch\\NATS-tss-v1_0-3ffb9-full", 'tss', fast_mode=True, verbose=True)
print(f"12th architecture string: {api.arch(12)}")

config = api.get_net_config(9, 'cifar10')
network = get_cell_based_tiny_net(config)

print(f"12th architecture source code: {network}")

info = api.get_more_info(10000, 'cifar10')
print(info)

params = api.get_net_param(9, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))
