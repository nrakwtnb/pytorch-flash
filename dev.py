

tensorboardX_flags = ['t', 'tf', 'tensorboard', 'tensorboardX']
visdom_flags = ['v', 'vis', 'visdom']


import functools

"""
ToDo
    * rename this wrapper fucntion
    * multi-gpu ( maybe nothing to do ? )
    * allow two input types : dict or single Tensor
    * ? attach loss_fn to model (to consider)
"""
def forward_wrap(func):
    @functools.wraps(func)
    def _forward_wrap(self, inputs, target_device=None, retain_comp_graph=True):
        # this is only valid for the case where model is put on a single device
        device = next(self.parameters()).device
        outputs = func(self, { k:v.to(device) for k,v in inputs.items() })
        if retain_comp_graph:
            outputs = outputs.detach()
        if target_device is None or target_device == device:# assume the single device for the model too
            return outputs
        else:
            # exception case needed if cannot be passed onto the device
            return { k:v.to(target_device) for k,v in outputs.items() }
    return _forward_wrap


def input_default_wrapper(batch):
    return {"x":batch[0]},{"y":batch[1]}




import torch

def get_model(config, model_generator, model_args={}):
    model = model_generator(**model_args)
    
    if config["device"]["num_gpu"] > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model.cuda(config["device"]["gpu"])
        config["device"]["name"] = device
    else:
        config["device"]["name"] = 'cpu'
    
    config["object"]["model"] = model



def wrap_metrics(func, get_y_values):
    def _wrap_metrics(results, *args, **kwargs):
        return func(*get_y_values(results, *args, **kwargs))
    return _wrap_metrics



# to rename the function later
def check(config):
    if "grad_accumulation_steps" in config["other"].keys():
        grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
        batch_size = config["train"]["batch_size"]
        train_batch_size = batch_size // grad_accumulation_steps
        assert train_batch_size * grad_accumulation_steps == batch_size
        config["train"]["train_batch_size"] = train_batch_size
    else:
        config["train"]["train_batch_size"] = config["train"]["batch_size"]
    
    if "object" not in config.keys():
        config["object"] = {}
        
    from collections import defaultdict
    config["object"].update({"metrics_log" : defaultdict(lambda :[])})



# default
def get_y_values(results):
    y_pred = results['outputs']['y']
    y_true = results['labels']['y']
    return y_pred, y_true.to(y_pred.device)



def setup_metrics(config, output_transform, loss_fn):
    from ignite.metrics import Loss, Accuracy
    
    loss = wrap_metrics(loss_fn, output_transform)
    config["object"]["loss"] = loss
    
    metrics={
        'accuracy': Accuracy(output_transform=get_y_values),
        'loss': Loss(loss_fn, output_transform=get_y_values)
    }
    config["object"]["metrics"] = metrics






from ignite.engine.engine import Engine

"""
    * package args as config
    * delete output_transform arg ?
"""
#def create_trainer(model, optimizer, loss_fn, device=None, non_blocking=False, data_loader=None,
#                   input_transform=input_default_wrapper, output_transform=lambda results: results["loss"].item(), **kwargs):
def create_trainer(model, optimizer, loss_fn, device=None, non_blocking=False, data_loader=None,
        input_transform=input_default_wrapper, output_transform=lambda results:results, **kwargs):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    grad_accumulation_steps = kwargs.get('grad_accumulation_steps', 1)

    if data_loader is not None:
        dataset_size = len(data_loader.dataset)
        num_train_batches = len(data_loader)
        train_batch_size = data_loader.batch_size
        assert (dataset_size - 1) // train_batch_size + 1 == num_train_batches, "Invalid data_loader"
        batch_size = train_batch_size * grad_accumulation_steps
        final_batch_iters = (dataset_size - 1) // batch_size * batch_size + 1
        final_batch_size = dataset_size - final_batch_iters + 1


    def _update(engine, batch):
        model.train()# needed every time ? After calling evaluator run, back to train mode for example...

        Is_update = False
        loss_weight = 1.
        if engine.state.iteration % grad_accumulation_steps == 0:
            loss_weight /= grad_accumulation_steps
            Is_update = True
        elif engine.state.iteration >= final_batch_iters:
            loss_weight = len(batch) / final_batch_size
            if engine.state.iteration == num_train_batches:
                Is_update = True

        inputs, labels = input_transform(batch)
        outputs = model(inputs)
        loss = loss_fn({"inputs" : inputs,"outputs":outputs, "labels":labels}) / grad_accumulation_steps
        loss.backward()

        if Is_update:
            optimizer.step()
            optimizer.zero_grad()

        return output_transform({"inputs":inputs, "outputs":outputs, "labels":labels, "loss":loss})

    return Engine(_update)


def create_evaluator(model, metrics={}, device=None, non_blocking=False,
                     input_transform=input_default_wrapper, output_transform=lambda results: results, **kwargs):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, labels = input_transform(batch)
            outputs = model(inputs)
        return output_transform({"inputs":inputs, "outputs":outputs, "labels":labels})

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine



def setup(config):
    objects = config["object"]
    model = objects["model"]
    optimizer = objects["optimizer"]
    loss = objects["loss"]
    device = config["device"]["name"]
    grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
    metrics = objects["metrics"]
    train_loader = objects["train_loader"]
    #metrics_log = objects["metrics_log"]
    
    trainer = create_trainer(model, optimizer, loss, device=device, data_loader=train_loader,
                             grad_accumulation_steps=grad_accumulation_steps)
    train_evaluator = create_evaluator(model, metrics=metrics, device=device)#, metrics_log=metrics_log)
    val_evaluator = create_evaluator(model, metrics=metrics, device=device)#, metrics_log=metrics_log)
    
    config["object"]["engine"] = {
        "trainer" : trainer,
        "train_evaluator" : train_evaluator,
        "val_evaluator" : val_evaluator
    }
    
    return trainer



from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
def create_default_events(config):
    vis_tool = config['other']['vis_tool']
    if vis_tool in tensorboardX_flags:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")
    elif vis_tool in visdom_flags:
        try:
            import visdom
        except ImportError:
            raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")
    else:
        from tqdm import tqdm


    log_dir = "tf_log"

    trainer = config["object"]["trainer"]
    grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
    train_loader = config["object"]["train_loader"]
    val_loader = config["object"]["val_loader"]
    train_evaluator = config["object"]["engine"]["train_evaluator"]
    val_evaluator = config["object"]["engine"]["val_evaluator"]


    num_train_batches = len(train_loader)# -> config
    log_interval = config["other"]["log_interval"]
    
    if vis_tool in tensorboardX_flags:
        def create_summary_writer(model, data_loader, log_dir):
            writer = SummaryWriter(logdir=log_dir)
            #data_loader_iter = iter(data_loader)
            #x, y = next(data_loader_iter)
            #try:
            #    writer.add_graph(model, x)
            #except Exception as e:
            #    print("Failed to save model graph: {}".format(e))
        return writer
        writer = create_summary_writer(model, train_loader, log_dir)
        config['object']['vis_tool'] = writer
    elif vis_tool in visdom_flags:
        vis = visdom.Visdom()
        def create_plot_window(vis, xlabel, ylabel, title):
            return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
        train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Loss')
        train_avg_loss_window = create_plot_window(vis, '#Iterations', 'Loss', 'Training Average Loss')
        train_avg_accuracy_window = create_plot_window(vis, '#Iterations', 'Accuracy', 'Training Average Accuracy')
        val_avg_loss_window = create_plot_window(vis, '#Epochs', 'Loss', 'Validation Average Loss')
        config['object']['vis_tool'] = vis
    else:
        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader),
            desc=desc.format(0)
        )
        config['object']['vis_tool'] = pbar

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        epoch = engine.state.epoch
        iter_ = engine.state.iteration
        iter_ = ((iter_ - 1) % num_train_batches) // grad_accumulation_steps + 1
        num_iter_per_epoch = (num_train_batches - 1) // grad_accumulation_steps + 1

        results = engine.state.output
        loss_val = results['loss'].item()
        print_message = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(epoch, iter_, num_train_batches, loss_val)
        if iter_ % log_interval == 0:
            if vis_tool in tensorboardX_flags:
                print(print_message, flush=True)
                writer.add_scalar("training/loss", loss_val, engine.state.iteration)
            elif vis_tool in visdom_flags:
                print(print_message, flush=True)
                vis.line(X=np.array([engine.state.iteration]),
                         Y=np.array([engine.state.output]), update='append', win=train_loss_window)
            else:
                pbar.desc = desc.format(loss_val)
                pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if vis_tool not in tensorboardX_flags and visdom_flags not in visdom_flags:
            pbar.refresh()
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print_logs(config, engine, metrics, phase='train')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print_logs(config, engine, metrics, phase='validation')
        if vis_tool not in tensorboardX_flags and visdom_flags not in visdom_flags:
            pbar.n = pbar.last_print_n = 0
    

    early_stopping_patience = 2
    model_checkpoint_save_interval = 1
    model_checkpoint_n_saved = 2
    model_name_prefix = "model"
    model_name = "model"
    model_dir = "models"
    trainer = config['object']['engine']['trainer']
    val_evaluator = config['object']['engine']['val_evaluator']
    model = config['object']['model']

    ES_handler = EarlyStopping(patience=early_stopping_patience, score_function=lambda engine:-engine.state.metrics['loss'], trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, ES_handler)
    MC_handler = ModelCheckpoint(model_dir, model_name_prefix, save_interval=model_checkpoint_save_interval, n_saved=model_checkpoint_n_saved, create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, MC_handler, {model_name : model})



def print_logs(config, engine, metrics, phase):
    vis_tool = config['other']['vis_tool']
    objects = config['object']
    metrics_log = objects.get("metrics_log", None)

    if metrics_log is not None:
        for name, value in metrics.items():
            metrics_log[f'{phase}/{name}'].append(value)

    epoch = engine.state.epoch
    
    print_message = f"{phase} results - Epoch: {epoch}" +\
        " ".join([f"{name}: {value:.4f}" for name, value in metrics.items() if isinstance(value, float)])
    
    if vis_tool in tensorboardX_flags:
        print(print_message, flush=True)
        for name, val in [("avg_loss", avg_nll), ("avg_accuracy", avg_accuracy)]:
            writer.add_scalar(f"{phase}/{name}", val, epoch)
    elif vis_tool in visdom_flags:
        print(print_message, flush=True)
        for val, win in [(avg_accuracy, val_avg_accuracy_window), (avg_nll, val_avg_loss_window)]:
            vis.line(X=np.array([epoch]), Y=np.array([val]), win=win, update='append')
    else:
        from tqdm import tqdm
        tqdm.write(print_message)

    
def close_logger():
    vis_tool = config['other']['vis_tool']
    if vis_tool in tensorboardX_flags:
        writer = config['object']['vis_tool']
        writer.close()
    elif vis_tool in visdom_flags:
        vis = config['object']['vis_tool']
        pass
    else:
        pbar = config['object']['vis_tool']
        pbar.close()


