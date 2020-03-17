
from flash.utils import _get_batchsize, _compute_start_indices, _partition_batch, _concat_results, _apply_transform
import numpy as np

class DataGateway():
    def __init__(self, default_input_size=None, max_batch_size=None, num_batch_division=1, input_transform=None):###
        assert isinstance(num_batch_division, int) and num_batch_division > 0
        

        self.set_up(default_input_size, max_batch_size, num_batch_division, input_transform)
        self.clear_cache()
    
    def set_up(self, default_input_size, max_batch_size, num_batch_division, input_transform):
        if input_transform is None:
            input_transform = lambda x:x
        self.default_input_size = default_input_size
        self.input_transform = input_transform
        self.max_batch_size = max_batch_size
        self.num_batch_division = num_batch_division
        
        #if self.default_input_size is not None and self.num_batch_division is not None:# require num_batch_division > 1 ?
        if self.default_input_size is not None:# require num_batch_division > 1 ?
            self.default_start_indices = _compute_start_indices(self.default_input_size, self.num_batch_division)
            self.default_batch_sizes = np.diff(self.default_start_indices)
            if self.max_batch_size is not None:
                assert self.default_batch_sizes.max() <= self.max_batch_size
        else:
            self.default_start_indices = None
            
        default_start_indices = self.default_start_indices
        default_batch_sizes = self.default_batch_sizes
        
        def divide_inputs(batch):
            inputs = input_transform(batch)
            if num_batch_division == 1:
                # to correct ...
                # assert input_size <= max_batch_size (if not None) and then divede ...
                return [inputs], [1.]
            
            input_size = _get_batchsize(inputs)
            if input_size == default_input_size:
                return _partition_batch(inputs, default_start_indices), default_batch_sizes / input_size
            else:
                if default_start_indices is not None:
                    start_indices = [ idx for idx in default_start_indices if idx < input_size ] + [input_size]
                elif max_batch_size is not None:
                    start_indices = np.arange(0, input_size, max_batch_size)
                else:
                    start_indices = _compute_start_indices(input_size, num_batch_division)
                portion_in_inputs = np.diff(start_indices) / input_size
                return _partition_batch(inputs, start_indices), portion_in_inputs
        self.divide_inputs = divide_inputs
    
    def clear_cache(self):
        self.outputs_stack = []
        self.loss_stack = []
    
    def get_inputs(self, inputs):
        # save partitioned_inputs if necessary
        #self.partitioned_inputs, self.portition_in_inputs = self.divide_inputs(self.inputs) ### remove self in the next line
        self.clear_cache()
        partitioned_inputs, portition_in_inputs = self.divide_inputs(inputs)
        #return zip(self.partitioned_inputs, self.portition_in_inputs)
        return zip(partitioned_inputs, portition_in_inputs)

    """
    def stack_inputs(self, obj, key, transform_info={}, **kwargs):
        if len(transform_info) > 0:
            obj = _apply_transform(obj, **transform_info)
        if key == 'label':
            self.labels_stack.append(obj)
        elif key == 'others':
            self.inputs_stack.append(obj)
        else:
            # unknown key
            raise NotImplementedError
            
    """
        
    #def stack_outputs(self, obj, key, transform_info={}, **kwargs):
    def stack_outputs(self, obj, obj_type, transform_info={}, **kwargs):
        if len(transform_info) > 0:
            obj = _apply_transform(obj, **transform_info)
        if obj_type == 'loss':
            self.loss_stack.append(obj)
        elif obj_type == 'outputs':
            self.outputs_stack.append(obj)
        else:
            # unknown key
            raise NotImplementedError
            
    def get_results(self):
        results = {}
        if len(self.labels_stack) > 0:
            labels = _concat_results(self.labels_stack)
            results.update({'labels':labels})
        if len(self.others_stack) > 0:
            others = sum(self.others_stack)
            results.update({'others':others})
        if len(self.outputs_stack) > 0:
            outputs = _concat_results(self.outputs_stack)
            results.update({'outputs':outputs})
        if len(self.loss_stack) > 0:
            loss = sum(self.loss_stack)
            results.update({'loss':loss})
        return results

from collections import defaultdict
class DataWarehouse():
    def __init__(self):
        self._reset()

    def _reset(self):
        self.data_stack = defaultdict(lambda :[])
        self.data_tree = None# NotImplemeted
    
    def stack(self, data, key):
        self.data_stack[key].append(data)
    
    def save(self, data, key):
        self.stack(data, key)
        raise NotImplementedError

class TrainDataServer():
    # Note : args are temporal ... (maybe remove ata_gateway arg... in the future)
    def __init__(self, data_gateway, data_warehouse, data_loader=None):
        self.data_gateway = data_gateway
        self.data_warehouse = data_warehouse
        self.data_loader = data_loader
        if data_loader is None:
            self.current_data_loader_name = "none"
            self.data_loader_info = {}
        else:
            self.current_data_loader_name = "default"
            self.data_loader_info = { self.current_data_loader_name : data_loader }
        
        self.current_buffer = defaultdict(lambda :{})#{}#data_gateway.init_buffer()
        
        self._reset_epoch()
        self._reset_iter()
        self._ready = False
    
    def register_data_loader(self, data_loader_dict):
        assert set(self.data_loader_info.keys()) & set(data_loader_dict.keys()) == set(), "Added data loaders overlap with the registed ones."
        self.data_loader_info.update(data_loader_dict)

    @property
    def data_loader_keys(self):
        return list(self.data_loader_info.keys())

    def set_data_loader(self, key=None):
        assert key in self.data_loader_info.keys(), f"No data loader named '{key}' is registed."
        self.data_loader = self.data_loader_info[key]
        self.current_data_loader_name = key
        self._ready = False

    #def __setattr__(self, name, value):   
    #def set_data_loader(self, data_loader):
    #    self.data_loader = data_loader
        
    def __iter__(self):
        self._reset_epoch()
        self._data_loader = iter(self.data_loader)
        self._ready = True
        return self

    def __next__(self):
        #self._ready = True
        self.inputs = next(self._data_loader)
        self.iter += 1
        return self.iter
        #return self.iter, self.inputs# new proposal
    
    def _reset_epoch(self):
        self.iter = 0
        self._ready = False

    def _reset_iter(self):
        self.outputs = {}
        self.loss = {}
        #self.outputs_stack = []
        #self.loss_stack = []

    #def _reset_inner_iter(self):

    def generate_inputs(self):# name : generate_inputs -> pull
        #if not self._ready:
        #    raise Exception
        self._reset_iter()
        #self._reset_inner_iter()
        return self.data_gateway.get_inputs(self.inputs)

    def push(self, obj, obj_type, key=None, transform_info={}, **kwargs):
        # remove key in args ?
        self.data_gateway.stack_outputs(obj, obj_type, transform_info=transform_info, **kwargs)

    #def register(self, name_for_register=None, Save_to_warehosue=False):
    def register(self, name_for_register=None, keys_to_save=set(), save_key_rule=lambda key:f'current/{key}', Save_all=False):
        """
            keys_to_save : iterator of str which can be converted into a set
        """
        results = self.data_gateway.get_results()
        if name_for_register is None:
            name_for_register = "model"# const var ? or allow None (in thi case, modify _convert_from_json_to_tree func)
        #keys_to_save = set(keys_to_save)
        
        for key, val in results.items():
            self.current_buffer[key].update({ name_for_register : val })
            if Save_all or key in keys_to_save:
                self.save(obj=val, key=save_key_rule(key))
            
        """
        #if Save_to_warehosue:
            #for key, val in self.current_buffer.items():
            for key in keys_to_save:
                val = self.current_buffer
                self.save(obj=val, key=f'current/{key}')
        """
    
    @property
    def results(self):# other than current buffer vase ? or rename ?
        return _convert_from_json_to_tree(self.current_buffer)
    
    def save(self, obj, key):
        self.data_warehouse.stack(obj, key)### temp
    
    #def push(self, obj, key, transform_type_info={}, Register=False, **kwargs):# write ?
    #def pull(self, obj, key, transform_type_info={}, Register=False, **kwargs):# read ?
    
    #def get_results(self):
    #    return { "inputs" : self.inputs, "outputs" : self.outputs, "loss" : self.loss }

### will move on to utils.py in the future

from collections import ChainMap

SEP = '/'

def _convert_from_json_to_tree(data, parent_key=""):
    if isinstance(data, dict):
        return dict(ChainMap(*[ _convert_from_json_to_tree(v, parent_key+SEP+k if len(parent_key) > 0 else k ) for k,v in data.items() ]))
    else:
        return { parent_key : data }
