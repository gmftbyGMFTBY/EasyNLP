import yaml

def load_config(args):
    model = args['model']
    config_path = f'config/{model}.yaml'
    print(f'[!] load configuration: {config_path}')
    configuration = yaml.load(open(config_path))[args['mode']]

    # base config
    base_configuration = yaml.load(open('config/base.yaml'))
    configuration.update(base_configuration)

    # load by lang
    args['lang'] = base_configuration['datasets'][args['dataset']]
    configuration['tokenizer'] = configuration['tokenizer'][args['lang']]
    if args['mode'] in ['train']:
        configuration['pretrained_model'] = configuration['pretrained_model'][args['lang']]
    return configuration

def load_base_config():
    config_path = f'config/base.yaml'
    print(f'[!] load base configuration: {config_path}')
    configuration = yaml.load(open(config_path))
    return configuration

