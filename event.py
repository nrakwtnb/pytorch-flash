

from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

tensorboardX_flags = ['t', 'tf', 'tensorboard', 'tensorboardX']
visdom_flags = ['v', 'vis', 'visdom']



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

    #trainer = config["object"]["trainer"]
    grad_accumulation_steps = config["other"]["grad_accumulation_steps"]
    train_loader = config["object"]["train_loader"]
    val_loader = config["object"]["val_loader"]
    train_evaluator = config["object"]["engine"]["train_evaluator"]
    val_evaluator = config["object"]["engine"]["val_evaluator"]
    trainer = config['object']['engine']['trainer']


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
        desc = "ITERATION - loss: "#{:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader)#,
            #desc=desc.format(0)
        )
        config['object']['vis_tool'] = pbar

    import torch
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        epoch = engine.state.epoch
        iter_ = engine.state.iteration
        iter_ = ((iter_ - 1) % num_train_batches) // grad_accumulation_steps + 1
        num_iter_per_epoch = (num_train_batches - 1) // grad_accumulation_steps + 1

        results = engine.state.output
        results_loss = results['loss']
        if isinstance(results_loss, torch.Tensor):
            loss_val = results_loss.item()
            print_message = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(epoch, iter_, num_train_batches, loss_val)
        else:
            if isinstance(results_loss, list):
                loss_val = { str(n):l.item() for n,l in enumerate(results_loss, 1) }
            elif isinstance(results_loss, dict):
                loss_val = { k:l.item() for k,l in results_loss.items() }
            else:
                assert False, 'Invalid type for loss'
            print_message = f"Epoch[{epoch}] Iteration[{iter_}/{num_train_batches}] Loss: "+"".join([ f"(k) {v:.2f}" for k,v in loss_val.items() ])

        if iter_ % log_interval == 0:
            if vis_tool in tensorboardX_flags:
                print(print_message, flush=True)
                writer.add_scalar("training/loss", loss_val, engine.state.iteration)
            elif vis_tool in visdom_flags:
                """
                    to correct in th future
                """
                print(print_message, flush=True)
                vis.line(X=np.array([engine.state.iteration]),
                         Y=np.array([engine.state.output]), update='append', win=train_loss_window)
            else:
                pbar.desc = print_message#desc.format(loss_val)
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
    
    if "handlers" in config.keys():
        handlers =  config['handlers']

        if 'early_stopping' in handlers.keys():
            hdl_early_stopping = handlers['early_stopping']
            patience = hdl_early_stopping['patience']
            if 'score_function' in hdl_early_stopping.keys():
                score_function = hdl_early_stopping['score_function']
            else:
                score_function = lambda engine:-engine.state.metrics['loss']
            ES_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
            val_evaluator.add_event_handler(Events.COMPLETED, ES_handler)

        if 'checkpoint' in handlers.keys():
            hdl_checkpoint = handlers['checkpoint']
            save_interval = hdl_checkpoint.get('save_interval',1)
            n_saved = hdl_checkpoint.get('n_saved', float('inf'))
            model_name_prefix = hdl_checkpoint.get("prefix", '')
            model_name = hdl_checkpoint.get("name", 'model')
            model_dir = hdl_checkpoint.get("save_dir", 'checkpoints')

            model = config['object']['model']
            MC_handler = ModelCheckpoint(model_dir, model_name_prefix, save_interval=save_interval, n_saved=n_saved, create_dir=True)
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


