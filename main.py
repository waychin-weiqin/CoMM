import argparse
import pandas as pd
from utils import *
from methods import com

import torch
from torch.utils import model_zoo
from network.wide_resnet import WideResNet
from network.resnet import resnet18, resnet50, model_urls

print(
    f"[INFO] Is CUDA available: {torch.cuda.is_available()} \n[INFO] Number of GPU detected: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def main(args):
    # Seet seed for reproducability
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    if args.source_dataset.upper() == "CIFAR-10":
        num_classes = 10
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, bias_last=True).to(device)
        state_dict = torch.load(args.model_path)
        _ = model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Model loaded from {args.model_path}, {_}")

    elif args.source_dataset.upper() == "CIFAR-100":
        num_classes = 100
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, bias_last=True).to(device)
        state_dict = torch.load(args.model_path)
        _ = model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Model loaded from {args.model_path}, {_}")

    elif args.source_dataset.upper() == "IMAGENET":

        assert args.network.find("resnet") != -1, f"[INFO] Model must be ResNet for ImageNet"

        num_classes = 1000

        # ResNet-18
        model = resnet18(pretrained=False, classes=num_classes).to(device)
        # args.lr = 0.005 * (args.tta_batchsize / 128)  # RN18
        if args.model_path is not None:
            state_dict = torch.load(args.model_path)
            _ = model.load_state_dict(state_dict, strict=True)
            print(f"[INFO] Model loaded from {args.model_path}, {_}")
        else:
            state_dict = model_zoo.load_url(model_urls['resnet18'])
            _ = model.load_state_dict(state_dict, strict=True)
            print(f"[INFO] Model loaded from Torchvision URL for ResNet18, {_}")

        else:
            raise Exception(f"[INFO] Architecture {args.network} not supported.")

    else:
        raise Exception(f"[INFO] Invalid dataset: {args.source_dataset}")

    # Get dataloaders (OOD)
    data_path = os.path.join(args.data_path, args.target_dataset.upper())
    if not os.path.exists(data_path):
        raise Exception(f"[INFO] Dataset not found at {data_path}")
    batch_size = args.tta_batchsize

    if args.source_dataset.upper().find("CIFAR") != -1:
        tta_train_loaders = prepare_cifar_loader(data_path=data_path, train=True, batch_size=batch_size,
                                                 severity=args.severity, corruptions=CORRUPTIONS)
        tta_test_loaders = prepare_cifar_loader(data_path=data_path, train=False, batch_size=1024,
                                                severity=args.severity, corruptions=CORRUPTIONS)

    elif args.source_dataset.upper().find("IMAGENET") != -1:
        tta_train_loaders = prepare_imagenet_loader(data_path=data_path, train=True, batch_size=batch_size,
                                                    severity=args.severity, corruptions=CORRUPTIONS)
        tta_test_loaders = prepare_imagenet_loader(data_path=data_path, train=False, batch_size=512,
                                                   severity=args.severity, corruptions=CORRUPTIONS)

    print("[INFO] Dataloaders ready")

    # Test-time training
    print(f"[INFO] Starting TTA")
    # TTA: train
    tta_error = dict()
    for domain in tta_train_loaders.keys():
        _ = model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Resetting model to original state. {_}")

        tr_loader = tta_train_loaders[domain][str(args.severity)]
        te_loader = tta_test_loaders[domain][str(args.severity)]

        print(f"----------  Corruption: {domain}, Severity: {args.severity}  ------------")

        if args.eval_before:
            # Compute before adaptation performance
            model.eval()
            before_loss, before_acc, before_cos_acc = test(model, te_loader, device)

        # Perform test-time adaptation
        best_error = 0
        model.train()
        model = com(model,
                    tr_loader,
                    args.criterion,
                    device,
                    lr=args.lr)

        # Compute after adaptation performance
        model.eval()
        after_loss, after_acc = test(model, te_loader, device)
        tta_error[domain] = (1 - after_acc) * 100

        if args.eval_before:
            # Print results
            print(f"Before Adaptation: Error: {1 - before_acc:.3%},  Cosine Error: {1 - before_cos_acc:.3%} \n"
                  f"After Adaptation: Error: {1 - after_acc:.3%}, Best Error: {best_error:.3%}")
        else:
            print(
                f"After Adaptation: Error: {1 - after_acc:.3%}, Best Error: {best_error:.3%}")
        print(f"------------------------------------------------------------------")
        print(" ")

    final_error = 0
    for domain in tta_error.keys():
        final_error += tta_error[domain]
    final_error /= len(tta_error.keys())
    print(f"[INFO] Final Accuracy: {final_error:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoMM Test-time Adaptation")
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default="/home/wei/data2/Dataset/cifar/", help='Path to data')
    parser.add_argument('--eval_before', action='store_true', default=False, help='Evaluate before adaptation')
    parser.add_argument('--source_dataset', type=str, default="cifar-10", help='Source dataset')
    parser.add_argument('--target_dataset', type=str, default="cifar-10-c", help='Target dataset')
    parser.add_argument('--criterion', type=str, default="cosine", help='Loss function to use')
    parser.add_argument('--network', default="wrn-40x2", type=str, help='Network architecture')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--tta_batchsize', default=128, type=int, help='Batch size for test-time training')
    parser.add_argument('--severity', default=5, type=int, help='Severity of corruption')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    args = parser.parse_args()

    if args.model_path == "None":
        args.model_path = None

    # print args
    for arg, value in vars(args).items():
        print(f"[INFO] {arg:<30}:  {str(value)}")
    print("--------------------  Initializing TTA  --------------------")
    main(args)
