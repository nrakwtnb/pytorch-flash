


import functools
import torch


DEBUG = True


### utils


"""

Idea:
    * update_info -> turn on as context ?
"""
"""
def forward_wrap_imp(func):
    @functools.wraps(func)
    def _forward_wrap(self, inputs, target_device=None, retain_comp_graph=True, update_info=None, **kwargs):
        # this is only valid for the case where model is put on a single device
        device = next(self.parameters()).device# self.device ...
        if update_info is not None:
            Is_updated = self.name in update_info.get('updated_models', [self.name])
            Is_computed = self.name in update_info.get('computed_models', [self.name])
            if not Is_computed:
                return {}
            if not Is_updated:
                retain_comp_graph = False
        # > outputs = func(self, { k:v.to(device) for k,v in inputs.items() })
        outputs = func(self, _apply_transform(inputs, device=device), **kwargs)
        # > if retain_comp_graph:
        # >    outputs = outputs.detach()
        if (target_device is None or target_device == device) and retain_comp_graph:# assume the single device for the model too
            return outputs
        else:
            # exception case needed if cannot be passed onto the device
            # > return { k:v.to(target_device) for k,v in outputs.items() }
            return _apply_transform(outputs, device=target_device, retain_comp_graph=retain_comp_graph)
    return _forward_wrap
"""


