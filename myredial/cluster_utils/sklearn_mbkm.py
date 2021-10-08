from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np

'''Mini-Batch KMeans'''

def kmeans(data, ncluster, batch_size):
    mbkm.MiniBatchKMeans(
        n_cluster=ncluster,
        batch_size=batch_size,
        random_state=9,
        verbose=True,
    )
    output = mbkm.fit(data)
    centers, labels = output.clueters_centers_, output.labels_
    return centers, labels

def cluster_on_context_and_response_space(args):
    res_embds, ctx_embds, ctexts, rtexts = [], [], [], []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                res_embd, ctx_embd, ctext, rtext = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt')
            except:
                break
            res_embds.append(res_embd)
            ctx_embds.append(ctx_embd)
            ctexts.extend(ctext)
            rtexts.extend(rtext)
    res_embds = np.concatenate(res_embds)    # [N, E]
    ctx_embds = np.concatenate(ctx_embds)    # [N, E]
    res_centers, res_labels = kmeans(res_embds, args['ncluster'], args['cluster_batch_size'])
    print(f'[!] over training the response semantic space')
    ctx_centers, ctx_labels = kmeans(res_embds, args['ncluster'], args['cluster_batch_size'])
    print(f'[!] over training the context semantic space')
    torch.save(
        (ctx_centers, ctx_labels, ctexts, res_centers, res_labels, rtexts), 
        f'{args["root_dir"]}/data/{args["dataset"]}/train_kmeans_ctx_res.pt'
    )


