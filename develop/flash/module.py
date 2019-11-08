
class ProcessManager():
    def __init__(self, model, loss_fn=None, optimizer=None, pre_operator=None, post_operator=None,
                 skip_condition=None, break_condition=None, process_name=None, num_iter=1, mode='train',
                 keep_outputs=lambda iter_, num_iter, state, name:(True, str(name)+f"{iter_}", {"retain_comp_graph":False}),
                 keep_loss=lambda iter_, num_iter, state:True):
        self.process_name = process_name
        self.num_iter = num_iter
        self.mode = mode
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.skip_condition = skip_condition
        self.break_condition = break_condition
        self.pre_operator = pre_operator
        self.post_operator = post_operator
        
        self.keep_loss = keep_loss
        self.keep_outputs = keep_outputs

    def set_data_server(self, data_server):
        self.data_server = data_server

    def __call__(self, inputs=None, no_grad=True, **kwargs):
        if no_grad:
            outputs = self.model(inputs, **kwargs)
        else:
            with torch.no_grad():
                outputs = self.model(inputs, **kwargs)
        return outputs

    def run(self, state=None, process_config={}):
        if self.skip_condition is not None:
            if self.skip_condition(self, state=state):
                return None
        
        self.model.train()
        #if mode == 'train':
        #    self.model.train()# needed every time ? After calling evaluator run, back to train mode for example...
        #elif mode == 'eval':
        #    self.model.eval()# needed every time ? After calling evaluator run, back to train mode for example...
        
        # allow num_iter to be infinity
        iter_ = 0
        while iter_ < self.num_iter:
            Break_iteration = self._run(iter_, state, process_config)
            if Break_iteration:
                break
            iter_ += 1

    def _run(self, iter_, state, process_config={}):
        # process_ocnfigは今のところ使用していない
        data_server = self.data_server
        
        if self.pre_operator is not None:
            # post num_iter as well ?
            self.pre_operator(self, state=state, iter_=iter_, data_server=data_server)
            
        Keep_outputs, output_name, outputs_type_info = self.keep_outputs(iter_, self.num_iter, state, self.process_name)
        Keep_loss = self.keep_loss(iter_, self.num_iter, state)
        for inputs, size_portion in data_server.generate_inputs(self.process_name, iter_):
            # add data_controller ? (inputsをdata_serverに要求)
            # modelはforward_wrap済み、と仮定
            outputs = self(inputs)
            
            if self.loss_fn is not None:
                data = {'inputs' : inputs, 'outputs' : outputs}
                loss = self.compute_gradient(data, size_portion)
                if Keep_loss:
                    data_server.stack_loss(loss)
            
            if Keep_outputs:
                data_server.stack_outputs(outputs, outputs_type_info)

        
        data_server.set_outputs()
        if self.loss_fn is not None:
            data_server.set_loss()

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.post_operator is not None:
            #post_operator(model=model, optimizer=optimizer, state=engine.state, iter_=iter_, outputs=outputs_stage, loss=loss_stage)
            self.post_operator(self, state=state, iter_=iter_, data_server=data_server)

        if self.break_condition is not None:
            return self.break_condition(self, state=state, iter_=iter_, data_server=data_server)
        else:
            return False

    def compute_gradient(self, data, weight, retain_comp_graph=False):
        #data = self.data_server.get_current_data()
        loss = self.loss_fn(data) * weight
        loss.backward()
        if not retain_comp_graph:
            loss = loss.detach()
        return loss






from flash.utils import _get_batchsize, _compute_start_indices, _partition_batch, _concat_results, _apply_transform
import numpy as np
import threading

class DataServer():
    __singleton = None
    __new_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        cls.__new_lock.acquire()
        if cls.__singleton == None:
            cls.__singleton = super(DataServer, cls).__new__(cls)
        cls.__new_lock.release()
        return cls.__singleton


    def __init__(self, default_input_size=None, max_batch_size=None, num_batch_division=1, input_transform=None):###
        assert isinstance(num_batch_division, int) and num_batch_division > 0
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
        
        def divide_inputs(inputs):
            if num_batch_division == 1:
                return [inputs], [1.]
            
            input_size = _get_batchsize(inputs)
            if input_size == self.default_input_size:
                return _partition_batch(inputs, self.default_start_indices), self.default_batch_sizes / input_size
            else:
                if self.default_start_indices is not None:
                    start_indices = [ idx for idx in self.default_start_indices if idx < input_size ] + [input_size]
                elif self.max_batch_size is not None:
                    start_indices = np.arange(0, input_size, self.max_batch_size)
                else:
                    start_indices = _compute_start_indices(input_size, num_batch_division)
                portion_in_inputs = np.diff(start_indices) / input_size
                return _partition_batch(inputs, start_indices), portion_in_inputs
        self.divide_inputs = divide_inputs
        
        self.outputs_stack = []
        self.loss_stack = []
   
    def set_up(self, batch):
        self.inputs = self.input_transform(batch)
        self.outputs = {}
        self.loss = {}
 
    def stack_outputs(self, outputs, outputs_type_info={}):
        if len(outputs_type_info) > 0:
            outputs = _apply_transform(outputs, **outputs_type_info)
        self.outputs_stack.append(outputs)

    def stack_loss(self, loss):
        self.loss_stack.append(loss)

    def set_outputs(self):
        if len(self.outputs_stack) > 0:
            if self.stage_name is not None:
                self.outputs.update({ self.stage_name : _concat_results(self.outputs_stack) })
            else:
                self.outputs.update( _concat_results(self.outputs_stack) )

    def set_loss(self):
        if len(self.loss_stack) > 0:
            ### to be modified
            self.loss.update({ self.stage_name : sum(self.loss_stack) })

    def generate_inputs(self, stage_name, iter_):
        self.stage_name = stage_name
        self.outputs_stack = []
        self.loss_stack = []
        # save partitioned_inputs if necessary
        self.partitioned_inputs, self.portition_in_inputs = self.divide_inputs(self.inputs)
        return zip(self.partitioned_inputs, self.portition_in_inputs)
    
    def get_results(self):
        return { "inputs" : self.inputs, "outputs" : self.outputs, "loss" : self.loss }


