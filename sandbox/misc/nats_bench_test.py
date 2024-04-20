import os
from nats_bench import create
from dotenv import dotenv_values
from xautodl.models import get_cell_based_tiny_net

os.environ["HOME"] = dotenv_values()["HOME"]

api = create(None, 'tss', fast_mode=True, verbose=True)
print(f"12th architecture string: {api.arch(12)}")

config = api.get_net_config(12, 'cifar10')
network = get_cell_based_tiny_net(config)

print(f"12th architecture source code: {network}")

info = api.get_more_info(10000, 'cifar10')
print(info)
