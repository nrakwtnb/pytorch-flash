### process.py temporal

from types import MethodType

class Process():
    def __init__(self, func=None):
        if func is not None:
            self.run = MethodType(func, self)
        #raise NotImplementedError
        
    def set_data_server(self, data_server):
        self.data_server = data_server
        
    # def set_event_manager
    
    # improve
    def __repr__(self):
        return self.process_name
        
    def run(self, state=None, process_config={}):
        raise NotImplementedError
        
# pytorch dependent
class ModelFlowProcess(Process):
    def __init__(self, model, loss_fn=None, optimizer=None, pre_operator=None, post_operator=None,
                 skip_condition=None, break_condition=None, process_name=None, num_iter=1, mode='train',
                 keep_outputs=lambda iter_, num_iter, state, name:(True, str(name)+f'{iter_}', {'retain_comp_graph':False}),
                 keep_loss=lambda iter_, num_iter, state:True):
        self.process_name = process_name# move into Process class ?
        self.num_iter = num_iter
        self.mode = mode# necessary ?
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.skip_condition = skip_condition
        self.break_condition = break_condition
        self.pre_operator = pre_operator
        self.post_operator = post_operator
        
        self.keep_loss = keep_loss
        self.keep_outputs = keep_outputs

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
        # Fix : role of process_config
        data_server = self.data_server
        params = { 'state' : state, 'iter_' : iter_, 'data_server' : data_server }
        
        if self.pre_operator is not None:
            # post num_iter as well ?
            #self.pre_operator(self, state=state, iter_=iter_, data_server=data_server)
            self.pre_operator(self, **params)
            
        Keep_outputs, output_name, outputs_type_info = self.keep_outputs(iter_, self.num_iter, state, self.process_name)
        Keep_loss = self.keep_loss(iter_, self.num_iter, state)
        for inputs, size_portion in data_server.generate_inputs():
            # add data_controller ? (inputsをdata_serverに要求)
            # modelはforward_wrap済み、と仮定
            outputs = self(inputs, **process_config.get("forward_args", {}))
            
            if self.loss_fn is not None:
                data = {'inputs' : inputs, 'outputs' : outputs}
                loss = self.compute_gradient(data, size_portion)
                if Keep_loss:
                    #data_server.stack_loss(loss)
                    data_server.push(loss, key='loss')
            
            if Keep_outputs:
                #data_server.stack_outputs(outputs, outputs_type_info)
                data_server.push(outputs, transform_info=outputs_type_info, key='outputs')

        
        data_server.register(name_for_register=self.process_name)
        #data_server.set_outputs()
        #if self.loss_fn is not None:
        #    data_server.set_loss()
        #data_server.keep()

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.post_operator is not None:
            self.post_operator(self, **params)

        if self.break_condition is not None:
            return self.break_condition(self, **params)
        else:
            return False

    def compute_gradient(self, data, weight, retain_comp_graph=False):
        #data = self.data_server.get_current_data()
        loss = self.loss_fn(data) * weight
        loss.backward()
        if not retain_comp_graph:
            loss = loss.detach()
        return loss

from collections import OrderedDict

class ProcessManager():
    def __init__(self, data_server, event_manager):
        self.data_server = data_server
        self.event_manager = event_manager
        self.process_info = OrderedDict()
        self.state = {}# class State in the future
        
        self.state["global/epoch"] = 1
        self.state["global/iteration"] = 1
    
    def register_process(self, process, process_name=None):
        # assert hasattr(process,run)
        # type name : str
        assert process_name not in self.process_info.keys()
        
        if process_name is None:
            process_name = f"process-{len(self.process_info)+1}"
        
        # (process, process_config)
        self.process_info.update({process_name : (process, {})})
        
    def register_multiple_processes(self, process_info):
        # assert hasattr(process,run)
        #assert process_name not in self.process_info.keys()
        
        # (process, process_config)
        # may change into list from dict because Process class has its name ...
        self.process_info.update(process_info)
        
    def __iter__(self):
        return enumerate(self.process_info.values(), 1)
        
    def _reset_state(self):
        self.state["run/epoch"] = 1
        self.state["run/iteration"] = 1
    
    def _set_data_server(self):
        for _, (process, _) in self:
            process.set_data_server(self.data_server)
    
    def run(self, epoch_loops=1):
        # ToDo: Consider the training restart case
        self._reset_state()
        state = self.state
        
        self._set_data_server()
        data_server = self.data_server
        event_manager = self.event_manager
        event_manager._set_state(state)
        
        event_manager.fire_events("run-start")
        for epoch in range(epoch_loops):
            event_manager.fire_events("epoch-start")
            
            for iteration in data_server:
                event_manager.fire_events("iter-start")
                #self.state.output = self._process_function(self, self.state.batch)

                for N, (process, process_config) in self:
                    process.run(state=state, process_config=process_config)
                event_manager.fire_events("iter-end")
                #results = data_server.get_results()
                self.state["run/iteration"] = iteration
                self.state["global/iteration"] += 1
                
            event_manager.fire_events("epoch-end")
            self.state["run/epoch"] = epoch
            self.state["global/epoch"] += 1
            # evnet on exceptions
        event_manager.fire_events("run-end")

from collections import defaultdict
class EventManager():
    def __init__(self, data_server):
        self.events = defaultdict(lambda :[])
        self.data_server = data_server

    def register_event(self, event_type, event_process):
        # assert hasattr(process,run)
        self.events[event_type].append((event_process, {}))
        
    def register_multiple_processes(self, process_info):
        raise NotImplementedError
        
    def __iter__(self):
        return iter([ (event_type, (process, config)) for event_type, event_process_info in self.events.items() for process, config in event_process_info ])
        
    #def _reset_state(self):
    #    self.state["run/epoch"] = 1
    #    self.state["run/iteration"] = 1
    
    def _set_data_server(self):
        for _, (process, _) in self:
            process.set_data_server(self.data_server)
            
    def _set_state(self, state):
        self.state = state
    
    def fire_events(self, event_type):
        for event_process, event_config in self.events.get(event_type, []):
            event_process.run(self.state, event_config)
