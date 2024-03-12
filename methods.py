import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import test

def com(net,
        train_loader,
        test_loader,
        criterion: str = "ce",
        device: torch.device = torch.device('cuda'),
        **kwargs):
    net = net.train()

    # Collect BN params (Borrowed from EATA)
    params = []
    names = []
    for nm, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    optimizer = torch.optim.SGD(params, lr=kwargs['lr'], momentum=0.9)

    best_error = 1
    accuracies = []
    entropies = []
    # use tqdm to show progress bar
    for ii, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        ##################
        ### TRAIN MODE ###
        ##################
        net = net.train()
        optimizer.zero_grad()

        if criterion == "cosine":
            # Extract features
            features = net.extract(images)
            features = F.adaptive_avg_pool2d(features, 1)  # B, C, 1, 1
            features = features.view(features.size(0), -1)  # B, C

            # Compute outputs (logits)
            cls_weight = net.fc.weight  # N_class, C
            outp = features @ cls_weight.t()  # B, N_class

            # Compute cosine between features and classifier weights
            fea_norm = torch.norm(features, dim=1, keepdim=True)  # B, 1
            cls_norm = torch.norm(cls_weight, dim=1, keepdim=True)  # N_class, 1

            cosine = outp / (fea_norm * cls_norm.t() + 1e-6)  # B, N_class
            cosine = (cosine + 1) / 2

            # Get cosine max based on cosine itself
            cosine_max = torch.max(cosine, dim=1)[0]  # B

            # CoM
            # loss_cosine = torch.arccos(cosine_max)

            # CoMM
            loss_cosine = cosine_max / torch.sum(cosine, dim=1)  # B
            loss_cosine = -torch.log(loss_cosine)

            loss = loss_cosine.mean()

        elif criterion == "entropy":
            outp = net(images)
            entropy = -torch.sum(torch.softmax(outp, dim=1) * torch.log_softmax(outp, dim=1), dim=1)  # B
            loss = entropy.mean()

        elif criterion == "spl":
            outp = net(images)
            pseudo_label = torch.softmax(outp, dim=1)
            pseudo_label = torch.argmax(pseudo_label, dim=1).clone().detach()
            loss = F.cross_entropy(outp, pseudo_label, label_smoothing=0.2)

        elif criterion == "hpl":
            outp = net(images)
            pseudo_label = torch.softmax(outp, dim=1)
            pseudo_label = torch.argmax(pseudo_label, dim=1).clone().detach()
            loss = F.cross_entropy(outp, pseudo_label)

        else:
            raise ValueError(f"Criterion not defined: {criterion}. Please select one from [cosine, entropy]")

        # Train model with included samples
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outp.argmax(dim=1) == labels).float().mean().item()
        accuracies.append(accuracy)

        ##################
        ### TEST MODE ###
        ##################
        # Compute accuracy on test set at every step
        net = net.eval()
        with torch.no_grad():
            loss, acc = test(net, test_loader, device)

        # Update best error
        if (1 - acc) < best_error:
            best_error = 1 - acc

        print(
            f"Step: {ii}, Test Loss: {loss:.3f}, Test Accuracy: {acc:.3%}")

    return best_error, net, accuracies, entropies
