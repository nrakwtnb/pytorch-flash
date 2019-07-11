
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

def _get_batchsize(tensors):
    if isinstance(tensors, torch.Tensor):
        return len(tensors)
    elif isinstance(tensors, dict):
        return _get_batchsize(list(tensors.values())[0])
    elif isinstance(tensors, list):
        return _get_batchsize(tensors[0])



class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).

    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[0]):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        average_loss = self._loss_fn(output)
        """
        if isinstance(output, tuple) and len(output) == 2:
            y_pred, y = output
            average_loss = self._loss_fn(y_pred, y)
        elif isinstance(output, dict):
            average_loss = self._loss_fn(output)
        else:
            y_pred, y, kwargs = output
            average_loss = self._loss_fn(y_pred, y, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')
        """

        N = _get_batchsize(output)
        #N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples


