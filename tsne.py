import argparse
from random import shuffle
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
import models_encoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def get_args_parser():
    parser = argparse.ArgumentParser('GCMAE feature representation visual', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--batch_size', default='128', type=int,
                        help='batch size')
    # * Finetuning params
    parser.add_argument('--random', default=False,
                        help='random init only')
                        ### mae
                        # camelyon/pre
                        # nctcrc/pre
                        ###gcmae
                        # camelyon/pre
                        # nctcrc/pre
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--save_path', default='')
    parser.add_argument('--data_path_val', default='', type=str,
                    help='dataset val path')
    
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--gpu_id', default=0, type=int,
                        help="the order of gpu")
    return parser
def main(args):
    torch.cuda.set_device(args.gpu_id)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # weak augmentation
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6790435, 0.5052883, 0.66902906], std= [0.19158737, 0.2039779, 0.15648715])])

    dataset_val = datasets.ImageFolder(args.data_path_val, transform=transform_val)
    print(dataset_val)


    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_encoder.__dict__[args.model](
        global_pool=args.global_pool,
    )

    if args.finetune and not args.random:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    evaluate(data_loader_val, model, device)

def evaluate(data_loader, model, device):
    t = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, n_iter=5000)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    output_full = []
    target_full = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            output_full += output.cpu().numpy().tolist()
        target_full += target.cpu().numpy().tolist()
    output_full = np.array(output_full)
    target_full = np.array(target_full)
    t = t.fit_transform(output_full)
    

    x_min, x_max = t.min(0), t.max(0)
    print("x_min:{}./n\
        x_max:{}".format(x_min, x_max))
    X_norm = (t - x_min) / (x_max - x_min)
    print("X_norm shape:{}".format(X_norm.shape))


    plt.figure(figsize=(16, 16))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(target_full[i]), color=plt.cm.Set1(target_full[i]), fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.save_path)
    plt.show()    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)