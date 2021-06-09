import yaml

def load_config(args):
    mode, model = args['mode'], args['model']
    config_path = f'config/{model}-{mode}.yaml'
    print(f'[!] load configuration: {config_path}')
    configuration = yaml.load(open(config_path))

    # base config
    base_configuration = yaml.load(open('config/base.yaml'))
    configuration.update(base_configuration)
    return configuration

def load_base_config():
    config_path = f'config/base.yaml'
    print(f'[!] load base configuration: {config_path}')
    configuration = yaml.load(open(config_path))
    return configuration

