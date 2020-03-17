### process.py temporal

from types import MethodType

import torch

class Process():
    def __init__(self, func=None):
        #self.process_name
        #self.run_name
        if func is not None:
            self.run = MethodType(func, self)
        #raise NotImplementedError
        
    def set_data_server(self, data_server):
        self.data_server = data_server
        
    # def set_event_manager
    
    # improve
    #def __repr__(self):
    #    return self.process_name
        
    def run(self, state=None, process_config={}):
        raise NotImplementedError
        
# pytorch dependent
class ModelFlowProcess(Process):
    def __init__(self, model, loss_fn=None, optimizer=None, pre_operator=None, post_operator=None,
                 skip_condition=None, break_condition=None, process_name=None, num_iter=1, mode='train',
                 keep_inputs=lambda iter_, num_iter, state, name:(True, str(name)+f'{iter_}', {'retain_comp_graph':False}),
                 keep_outputs=lambda iter_, num_iter, state, name:(True, str(name)+f'{iter_}', {'retain_comp_graph':False}),
                 keep_loss=lambda iter_, num_iter, state:True):
        """
            if mode == 'eval':
                assert loss == None and optimizer == None
        """


        self.process_name = process_name# move into Process class ?
        self.run_name = "0"
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
        self.keep_inputs = keep_inputs
        self.keep_outputs = keep_outputs

    def __call__(self, inputs=None, no_grad=True, **kwargs):
        if no_grad:
            outputs = self.model(inputs, **kwargs)
        else:
            with torch.no_grad():
                outputs = self.model(inputs, **kwargs)
        return outputs

    def run(self, state=None, process_config={}):
        self.run_name = int(self.run_name) + 1
        if self.skip_condition is not None:
            if self.skip_condition(self, state=state):
                return None
        
        #self.model.train()
        if self.mode == 'train':
            self.model.train()# needed every time ? After calling evaluator run, back to train mode for example...
        elif self.mode == 'eval':
            self.model.eval()# needed every time ? After calling evaluator run, back to train mode for example...
        
        # allow num_iter to be infinity
        iter_ = 0
        while iter_ < self.num_iter:
            # move iter_ into state ???
            # maybe should not do because iter_ is not referred outside loop ...
            Break_iteration = self._run(iter_, state, process_config)
            if Break_iteration:
                break
            iter_ += 1

    def _run(self, iter_, state, process_config={}):
        # Fix : role of process_config
        data_server = self.data_server
        params = { 'process_name' : self.process_name, 'state' : state, 'iter_' : iter_, 'num_iter' : self.num_iter, 'data_server' : data_server, **process_config }
        
        if self.pre_operator is not None:
            #self.pre_operator(self, state=state, iter_=iter_, data_server=data_server)
            self.pre_operator(self, **params)
            
        Keep_inputs, input_name, inputs_type_info = self.keep_inputs(iter_, self.num_iter, state, self.process_name)
        Keep_outputs, output_name, outputs_type_info = self.keep_outputs(iter_, self.num_iter, state, self.process_name)
        Keep_loss = self.keep_loss(iter_, self.num_iter, state)
        for inputs, size_portion in data_server.generate_inputs(Keep_inputs, input_name, inputs_type_info):
            #print(len(inputs['x']))
            # add data_controller ? (inputsをdata_serverに要求)
            # modelはforward_wrap済み、と仮定
            if self.mode == 'train':
                outputs = self(inputs, **process_config.get("forward_args", {}))
            elif self.mode == 'eval':
                with torch.no_grad():
                    outputs = self(inputs, **process_config.get("forward_args", {}))
            
            if self.loss_fn is not None:
                data = {'inputs' : inputs, 'outputs' : outputs}
                loss = self.compute_gradient(data, size_portion)
                if Keep_loss:
                    #data_server.stack_loss(loss)
                    #data_server.push(loss, key='loss')
                    data_server.push(loss, obj_type='loss')

            #if Keep_inputs:
            #    data_server.push(inputs, transform_info=inputs_type_info, key=input_name)#'inputs')
            
            if Keep_outputs:
                #data_server.stack_outputs(outputs, outputs_type_info)
                #data_server.push(outputs, transform_info=outputs_type_info, key=output_name)#'outputs')
                data_server.push(outputs, obj_type='outputs', transform_info=outputs_type_info, key=output_name)#'outputs')

        
        data_server.register(name_for_register=self.process_name, **process_config.get("output_save_info", {}))
        #print(self.process_name, data_server.current_data_loader_name, process_config)
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

class ProcessManager(Process):
    def __init__(self):
        pass

    def _set_data_server(self):
        for _, (process, _) in self:
            process.set_data_server(self.data_server)

    def _set_state(self, state):
        self.state = state

    def _reset_state(self):
        raise NotImplementedError

from collections import OrderedDict

class ModelFlowManager(ProcessManager):
    def __init__(self, data_server, event_manager=None):
        self.data_server = data_server# self.set_data_server(data_server)
        self.event_manager = event_manager
        # temporal
        self.fire_events = event_manager.fire_events if event_manager is not None else lambda x:None
        #print(self.fire_events)
        self.process_info = OrderedDict()
        self.state = {}# class State in the future
        
        self.state["global/epoch"] = 1
        self.state["global/iteration"] = 1
    
    def register_process(self, process, process_name=None, process_config={}):
        # type name : str
        assert hasattr(process, 'run')

        if process_name is None:
            process_name = f"process-{len(self.process_info)+1}"
        
        if hasattr(process, 'name'):
            assert process_name == getattr(process, 'name', process_name)
        else:
            setattr(process, 'name', process_name)
        assert process_name not in self.process_info.keys()
        
        # (process, process_config)
        self.process_info.update({process_name : (process, process_config)})
        
    def register_multiple_processes(self, process_info):
        raise NotImplementedError
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

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager
        self.fire_events = event_manager.fire_events
    
    def run(self, state=None, process_config=dict(num_epochs=1)):
        num_epochs = process_config['num_epochs']
        #init_epoch = self.state["global/epoch"]#update_only
        data_loader_name = process_config.get('data_loader_name', None)## eval

        # ToDo: Consider the training restart case
        self._reset_state()
        state = self.state if state is None else state
        
        self._set_data_server()
        data_server = self.data_server
        if self.event_manager is not None:
            event_manager = self.event_manager
            event_manager._set_state(state)

        if data_loader_name is not None:### eval
            data_server.set_data_loader(data_loader_name)### eval
        
        #event_manager.fire_events("run-start")
        self.fire_events("run-start")
        for epoch in range(1, num_epochs+1):# modified
            #event_manager.fire_events("epoch-start")
            self.fire_events("epoch-start")
            
            for iteration in data_server:
                #event_manager.fire_events("iter-start")
                self.fire_events("iter-start")
                #self.state.output = self._process_function(self, self.state.batch)

                for N, (process, process_config_) in self:
                    print(N, process_config, process_config_)
                    process_config_ = {**process_config, **process_config_}
                    process.run(state=state, process_config=process_config_)
                #event_manager.fire_events("iter-end")
                self.fire_events("iter-end")
                #results = data_server.get_results()
                self.state["run/iteration"] = iteration
                self.state["global/iteration"] += 1
                
            #event_manager.fire_events("epoch-end")
            self.fire_events("epoch-end")
            self.state["run/epoch"] = epoch
            self.state["global/epoch"] += 1
            # evnet on exceptions
        #event_manager.fire_events("run-end")
        self.fire_events("run-end")

from collections import defaultdict
class EventManager(ProcessManager):
    def __init__(self, data_server):
        self.events = defaultdict(lambda :[])
        self.data_server = data_server

    def register_event(self, event_type, event_process, event_config):
        # assert hasattr(process,run)
        self.events[event_type].append((event_process, event_config))
        
    def register_multiple_processes(self, process_info):
        raise NotImplementedError
        
    def __iter__(self):
        return iter([ (event_type, (process, config)) for event_type, event_process_info in self.events.items() for process, config in event_process_info ])
        
    #def _reset_state(self):
    #    self.state["run/epoch"] = 1
    #    self.state["run/iteration"] = 1
    
    def fire_events(self, event_type):
        for event_process, event_config in self.events.get(event_type, []):
            event_process.run(self.state, event_config)
