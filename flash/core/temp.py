
num_train_batches = len(ds.data_loader)
grad_accumulation_steps = 1
log_interval = 2

def get_loss_from_results(results):
    return results['loss/']

import torch
def log_training_loss(self, state, process_config={}):
    """
        ### ToDo : check the following process
    """
    epoch = state["global/epoch"]
    iter_ = state["run/iteration"]
    iter_ = ((iter_ - 1) % num_train_batches) // grad_accumulation_steps + 1
    #num_iter_per_epoch = (num_train_batches - 1) // grad_accumulation_steps + 1

    if iter_ % log_interval == 0:
        results_loss = get_loss_from_results(self.data_server.results)
        #results_loss = results['loss']
        if isinstance(results_loss, torch.Tensor):
            loss_val = results_loss.item()
            print_message = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}".format(epoch, iter_, num_train_batches, loss_val)
        else:
            if isinstance(results_loss, list):
                loss_val = { f"({n})" if n != "" else "":l.item() for n,l in enumerate(results_loss, 1) }
            elif isinstance(results_loss, dict):
                loss_val = { k:l.item() for k,l in results_loss.items() }
            else:
                assert False, 'Invalid type for loss'
            print_message = f"Epoch[{epoch}] Iteration[{iter_}/{num_train_batches}] Loss: "+"".join([ f" {k} = {v:.4f} " for k,v in loss_val.items() ])
            
        print(print_message)
