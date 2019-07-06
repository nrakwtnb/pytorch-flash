
from ignite.engine.engine import Engine
from utils import input_default_wrapper


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
        loss = loss_fn({"inputs" : inputs,"outputs":outputs, "labels":labels}) * loss_weight
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

