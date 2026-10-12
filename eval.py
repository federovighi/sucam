import torch.nn.functional
from tools.viz import *
from train import *
import seaborn as sns

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

torch.manual_seed(0)
np.random.seed(0)

torch.set_printoptions(precision=10)


def eval(config, set, split, dataroot):
    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    '''     Yaw temporarly removed
    if config['backbone'] == 'lss':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180
    '''

    loader = datasets[config['dataset']](
        False, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
        #yaw=yaw
    )

    print(f"Using set: {set}")

    model = Model(
    config['gpus'],
    backbone=config['backbone'],
    loss_type='focal',
    use_dep=False,
    use_seg=False,
)

    if config['type'] == 'ensemble':
        state_dicts = [torch.load(path) for path in config['ensemble']]
        model.load(state_dicts)
    else:
        model.load(torch.load(config['pretrained']))

    model.eval()

    print("--------------------------------------------------")
    print(f"Running eval on {split}")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Loader: {len(loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Pretrained: {config['pretrained']} ")
    print("--------------------------------------------------")

    os.makedirs(config['logdir'], exist_ok=True)

    preds, labels, val_loss = run_loader(model, loader, config)

    return preds, labels, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-d', '--dataset', default='nuscenes', type=str)
    parser.add_argument('--type', default='single', type=str)
    parser.add_argument('--backbone', default='sucam', type=str)
    parser.add_argument('--gt_depth', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('-l', '--logdir', default='./test/eval_results', type=str)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument( '--split', default='trainval', required=False, type=str)
    parser.add_argument('-s', '--set', default='val', required=False, type=str)
    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--ensemble', nargs='+', required=False, type=str)
    parser.add_argument('-m', '--metric', default="rocpr", required=False)
    parser.add_argument('-r', '--save', default=False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pseudo', default=False, action='store_true')
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    print(f"Using config {args.config}")
    config = get_config(args)

    dataroot = f"../data/{config['dataset']}"
    split, metric, set = args.split, args.metric, args.set

    
    preds, labels, val_loss = eval(config, set, split, dataroot)

    _, _, iou = get_iou(preds, labels)

    print(f"Validation IoU: {iou}")
    print(f"Validation Loss: {val_loss:.6f}")

    # Save final metrics
    os.makedirs(config['logdir'], exist_ok=True)

    with open(os.path.join(config['logdir'], "metrics.txt"), "w") as f:
        f.write(f"Validation IoU: {iou}\n")
        f.write(f"Validation Loss: {val_loss:.6f}\n")
