import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader
import numpy as np
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def accuracy_from_loader(algorithm, loader, weights, is_test, inout, name, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()
    #print(name)
    for i, batch in enumerate(loader):

        #if is_test:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()
        
        # # checkpoint selection
        # elif not is_test and inout == 'out':
        #     x1 = batch["x1"].to(device)
        #     x2 = batch["x2"].to(device)
        #     y = batch["y"].to(device)

        #     with torch.no_grad():
        #         x1_logits = algorithm.predict(x1)
        #         x2_logits = algorithm.predict(x2)
        #         logits = x1_logits
        #         loss1 = F.cross_entropy(x1_logits, y).item()
        #         loss2 = F.cross_entropy(x2_logits, y).item()
        #         prob1 = F.softmax(x1_logits, dim=1)
        #         prob2 = F.softmax(x2_logits, dim=1)
        #         #loss = (loss1+loss2)/2.0 + 100* F.kl_div(prob2.log(), prob1).item()
        #         #loss = (loss1+loss2)/2.0 + 0.1*F.kl_div(prob2.log(), prob1, reduction='sum').item()
        #         #loss = 0.6*loss1+0.3*loss2+0.1*F.kl_div(prob2.log(), prob1, reduction='sum').item()
        #         #loss = 0.5*loss1+0.5*F.kl_div(prob2.log(), prob1).item()
        #         loss = loss1


        B = len(y)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(y))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(y)]
            weights_offset += len(y)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss


def accuracy(algorithm, loader_kwargs, weights, is_test, inout, name, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, is_test, inout, name, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm, loader_kwargs, weights, is_test, inout, name, debug=self.debug)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
