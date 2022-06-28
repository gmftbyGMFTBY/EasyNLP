import os
import subprocess
import json
from copy import deepcopy


def read_general_config(filename):
    with open(filename) as fp:
        config = json.load(fp)
    return config


def read_jizhi_config():
    nodes_list = [ip.split(':')[0]
                  for ip in os.environ['NODE_IP_LIST'].split(',')]
    master_ip = os.environ['LOCAL_IP'].strip()
    # print(nodes_list, master_ip)
    return {
        "nodes": nodes_list,
        "master": master_ip,
    }


def read_env():
    env = os.environ
    return env


def start(_env, _distributed_config, _my_config, ip, rank, remote=True):
    # env = deepcopy(_env)
    distributed_config = deepcopy(_distributed_config)
    my_config = deepcopy(_my_config)

    cmd_str = "python3 -m torch.distributed.launch --nproc_per_node=8 "
    for k, v in distributed_config.items():
        assert k.startswith('-')
        if not v == "":
            v = "'{}'".format(v)
        cmd_str += " " + k + "=" + v + " "

    cmd_str += " --node_rank={} ".format(rank)
    # cmd_str += " train.py "
    cmd_str += " train_long.py "

    for k, v in my_config.items():
        assert k.startswith('-')
        if not v == "":
            v = "'{}'".format(v)
        cmd_str += " " + k + " " + v + " "

    if remote:
        cmd_str = "ssh {} \"export NCCL_SOCKET_IFNAME=eth1; export NCCL_IB_DISABLE=1; cd /jizhi/jizhi2/worker/trainer; {}\"".format(
            ip, cmd_str)
        print(cmd_str)
        # return
        subprocess.Popen(cmd_str, shell=True)
    else:
        cmd_str = "export NCCL_SOCKET_IFNAME=eth1; export NCCL_IB_DISABLE=1; cd /jizhi/jizhi2/worker/trainer; {}".format(
            cmd_str)
        print(cmd_str)
        # return
        subprocess.call(cmd_str, shell=True)


def main(config_file):
    env = read_env()

    """build distributed config"""
    jizhi_config = read_jizhi_config()
    distributed_config = dict()
    distributed_config['--nnodes'] = str(len(jizhi_config['nodes']))
    distributed_config['--master_addr'] = jizhi_config['master']
    distributed_config['--master_port'] = "8081"

    """build custom config"""
    my_config = read_general_config(config_file)

    for i, ip in enumerate(jizhi_config['nodes']):
        if ip == jizhi_config['master']:
            continue
        start(env, distributed_config, my_config, ip, i, True)
    start(env, distributed_config, my_config,
          jizhi_config['master'], 0, False)

if __name__ == "__main__":
    config_file = 'distributor/config.json'
    main(config_file)
