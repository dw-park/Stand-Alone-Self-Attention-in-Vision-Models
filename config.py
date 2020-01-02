import argparse

def get_args():

    parser = argparse.ArgumentParser('Standalone Self-attention in Vision Models')

    parser.add_argument("--dataset", type=str, default='CIFAR10', help="CIFAR10")
    parser.add_argument("--model", type=str, default='ResNet26', help="Resnet26, ResNet38, ResNet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_worker", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stem", type=str, default="SA", help="SA, conv")
    parser.add_argument("--gpus", type=int, nargs="+", default=0)
    parser.add_argument("--data_aug", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="resnet_self_attention")
    args = parser.parse_args()

    return args
