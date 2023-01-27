

if __name__ == "__main__":
    for mode in ['train', 'test']:
        source_path = f'{mode}/source.txt'
        target_path = f'{mode}/target.txt'

        with open(source_path) as f:
            sources = [line.strip() for line in f.readlines() if line.strip()]

        with open(target_path) as f:
            targets= [line.strip() for line in f.readlines() if line.strip()]

        assert len(sources) == len(targets)

        with open(f'{mode}.txt', 'w') as f:
            for s, t in zip(sources, targets):
                s_l = s.split('|||')
                t_ls = t.split('|||')

                for i, t_l in enumerate(t_ls):
                    label = '1' if i == 0 else '0'
                    dialog = s_l + [t_l]
                    string = label + '\t' + '\t'.join(dialog) + '\n'
                    f.write(string)
