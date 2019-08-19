
import sys
sys.path.append('../../pytorch-flash/')

import os

save_dir = './tets_gcn'
output_save_dir = os.path.join(save_dir, 'outputs')
model_save_dir = os.path.join(save_dir, 'models')

config = {
    "device" : {
        "num_gpu" : 0,
        "gpu" : 0,
    },
    "train" : {
        "epochs" : 5,#1,
        "batch_size" : 32,#8,
    },
    "handlers" : {
        "early_stopping" : {
            "patience" : 5,
        },
        "checkpoint" : {
            "prefix" : 'gnn',
            'save_dir' : model_save_dir,
            'target_models' : ['classifier'],
        },
    },
    "others" : {
        "save_dir" : save_dir,
        "grad_accumulation_steps" : 1,
        "eval_batch_size" : 32,
        "log_interval" : 5,#10,
        "vis_tool" : "None",
    }
}

from manager import TrainManager
manager = TrainManager()
manager.set_config(config)


# ### data setup
import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader

# Create training and test sets.
train_dataset = MiniGCDataset(320, 10, 20)
val_dataset = MiniGCDataset(80, 10, 20)
train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=config['others']['eval_batch_size'], shuffle=False, collate_fn=collate)

manager.set_dataloader(train_loader=train_loader, val_loader=val_loader)


# ### model setup
from utils import forward_wrap

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    @forward_wrap
    def forward(self, inputs):
        g = inputs['graph']
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        y = self.classify(hg)
        return {'y':y}

import torch.optim as optiml
model = Classifier(1, 256, train_dataset.num_classes)

manager.add_model('classifier', model)

# ### optimizer setup
from optimizers import get_optimzier

optimizer_info = {
    "name" : "Adam",
    "args" : {
        "lr" : 0.001,
        #"betas" : (0.5, 0.999)
    },
}

optimizer = get_optimzier(optimizer_info, model)
manager.add_optimizer('model', optimizer)

CEloss = nn.CrossEntropyLoss()
def loss(results):
    y_pred = results['outputs']['y']
    y_true = results['inputs']['y'].to(y_pred.device)
    return CEloss(y_pred, y_true)

manager.add_loss_fn('classifier', loss)


# ### update & evaluate stages
update = {'model' : 'classifier', 'loss_fn' : 'classifier', 'optimizer' : 'model'}
manager.add_update_info(**update)

evaluate = {'model' : 'classifier'}
manager.add_evaluate_info(**evaluate)

manager.set_objects()

target_metrics = {'loss', 'accuracy', 'precision', 'recall', 'F1'}
manager.setup_metrics(target_metrics=target_metrics, target_loss_fn_names=None)

def input_transform(batch):
    return { 'graph': batch[0], 'y' : batch[1] }

trainer_args = {'input_transform':input_transform}
evaluator_args = {'input_transform':input_transform}
manager.setup_engines(trainer_args, evaluator_args)
manager.create_default_events()

manager.run()
manager.close()

