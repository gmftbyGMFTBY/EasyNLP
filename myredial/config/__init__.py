import yaml

def load_config(args):
    model = args['model']
    config_path = f'config/{model}.yaml'
    print(f'[!] load configuration: {config_path}')
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)[args['mode']]

    # base config
    base_configuration = load_base_config()
    configuration.update(base_configuration)

    # load by lang
    args['lang'] = base_configuration['datasets'][args['dataset']]
    configuration['tokenizer'] = configuration['tokenizer'][args['lang']]
    if args['mode'] in ['train']:
        configuration['pretrained_model'] = configuration['pretrained_model'][args['lang']]
    return configuration

def load_base_config():
    config_path = f'config/base.yaml'
    with open(config_path) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print(f'[!] load base configuration: {config_path}')
    return configuration

