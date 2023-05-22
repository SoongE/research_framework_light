import pandas
import torch
import torchmetrics.functional as TMF
from torch import distributed as dist


def all_gather(x):
    """Collect value to local rank zero gpu
    :arg
        x(tensor): target
    """
    if dist.is_initialized():
        dest = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(dest, x)
        return torch.cat(dest, dim=0)
    else:
        return x


def all_gather_with_different_size(x):
    """all gather operation with different sized tensor
    :arg
        x(tensor): target
    (reference) https://stackoverflow.com/a/71433508/17670380
    """
    if dist.is_initialized():
        local_size = torch.tensor([x.size(0)], device=x.device)
        all_sizes = all_gather(local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=x.device, dtype=x.dtype)
            x = torch.cat((x, padding))

        all_gathered_with_pad = all_gather(x)
        all_gathered = []
        ws = dist.get_world_size()
        for vector, size in zip(all_gathered_with_pad.chunk(ws), all_sizes.chunk(ws)):
            all_gathered.append(vector[:size])

        return torch.cat(all_gathered, dim=0)
    else:
        return x


def compute_metrics(probs, labels, task, num_classes, average='micro'):
    metric_names = ['accuracy', 'auroc', 'f1_score', 'specificity', 'recall', 'precision']

    if isinstance(probs, (list, tuple)):
        probs = torch.concat(probs, dim=0).detach().clone()
        labels = torch.concat(labels, dim=0).detach().clone()

    probs = all_gather_with_different_size(probs)
    labels = all_gather_with_different_size(labels)

    if task == 'binary':
        probs = probs.squeeze(1)

    metrics = list(TMF.__dict__[m](
        probs, labels, task, average='macro' if m == 'auroc' else average,
        num_classes=num_classes, num_labels=num_classes
    ) for m in metric_names)

    metric_names.append('confusion_matrix')
    metrics.append(
        TMF.__dict__['confusion_matrix'](probs, labels, task, num_classes=num_classes, num_labels=num_classes))

    return metrics, metric_names


def print_metrics(metrics, metric_names, average, epoch=0):
    space = 12
    num_metric = 1 + len(metrics)
    print('-' * space * num_metric)
    print(("{:>12}" * num_metric).format('Stage', *metric_names))
    print('-' * space * num_metric)
    if average == 'none':
        print(f"{f'VALID({epoch})':>{space}}" + "".join([f"{m.mean():{space}.4f}" for m in metrics]))
    else:
        print(f"{f'VALID({epoch})':>{space}}" + "".join([f"{m:{space}.4f}" for m in metrics]))
    print('-' * space * num_metric)

    # if average == 'none':
    #     for name, m in zip(metric_names, metrics):
    #         print(f'{name:>{space}} | ', end='')
    #         print(" ".join([f'{item:.3f}' for item in m.tolist()]))
    #     print('-' * space * num_metric)


@torch.no_grad()
def evaluate_performance(model, n_ff, loader, num_classes, device, task='multilabel', average='micro', verbose=False):
    probs = list()
    labels = list()
    model.eval()

    for _ in range(n_ff):
        for data in loader:
            x, y = map(lambda x: x.to(device), data)
            prob = model(x)
            probs.append(prob.detach().cpu())
            labels.append(y.detach().cpu())

    metrics, metric_names = compute_metrics(probs, labels, task, num_classes, average)

    if verbose:
        print_metrics(metrics, metric_names, average)
    df = pandas.DataFrame(metrics, index=metric_names).astype("float")
    df.to_csv('performance.csv')

    return metrics, metric_names
