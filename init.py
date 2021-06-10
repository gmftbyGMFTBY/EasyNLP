from config import load_base_config
import os, sys, ipdb

if __name__ == "__main__":
    config = load_base_config()

    datasets = config['datasets']
    models = config['models']
    root_dir = config['root_dir']

    for folder in ['rest', 'ckpt']:
        path = f'{root_dir}/{folder}'
        ipdb.set_trace()
        if not os.path.exists(path):
            os.mkdir(path)
        for dataset in datasets:
            subpath = f'{path}/{dataset}'
            if not os.path.exists(subpath):
                os.mkdir(path)
            for model in models:
                subsubpath = f'{path}/{model}'
                if os.path.exists(subsubpath):
                    os.mkdir(subsubpath)
    print(f'[!] init the folder under the {root_dir} over')
