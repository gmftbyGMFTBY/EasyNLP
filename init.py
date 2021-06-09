from config import load_base_config
import os, sys, ipdb

if __name__ == "__main__":
    config = load_base_config()

    datasets = config['datasets']
    models = config['models']
    root_dir = config['root_dir']

    for folder in ['rest', 'ckpt']:
        for dataset in datasets:
            if not os.path.exists(f'{root_dir}/{dataset}'):
                os.mkdir(f'{root_dir}/{dataset}')
            for model in models:
                path = f'{root_dir}/{dataset}/{model}'
                if os.path.exists(path):
                    os.mkdir(path)
    print(f'[!] init the folder under the {root_dir} over')
