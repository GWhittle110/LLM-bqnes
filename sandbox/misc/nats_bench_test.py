import os
from nats_bench import create
from dotenv import dotenv_values
from xautodl.models import get_cell_based_tiny_net
from random import sample
from yaml import dump

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

# Identify which networks we have parameters for
networks = [int(filename.split(".")[0])
            for filename in os.listdir(dotenv_values()["HOME"] + "\\.torch\\NATS-tss-v1_0-3ffb9-full")
            if not filename.startswith("meta")]

# Random sample 100 networks
candidates = sample(networks, 75)

# Save to yaml
with open('nats_indexes.yaml', 'w+') as f:
    dump({"nats_indexes": candidates}, f)
