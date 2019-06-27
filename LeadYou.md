### ToDo

* model
	+ allow model to tell what data it needs
	+ Ex.
		- > my_model.show_input_info
		- "images" : FloatTensor, size=(*,3,224,224)
		- "z_hidden" : FloatTensor, size=(*,10)
	+ input wrapper
		- See `dev.py`

* test
	+ multi gpu

* training
	+ modify iterations when grad_accumulation_steps > 1

* config

* metrics
	+ prepare many metrics (make it easy to logs ...)

* event setting
	+ each some couples of iterations
	+ evaluation (train dataset) at each epoch
	+ evaluation (validation dataset) at each epoch
	+ optimizer.grad_zero() at epoch starts

* visualization
	+ test : isdom
	+ add : tensorwatch

* backend integration
	+ compatibility with chainer

#### In future

* inference

* deploy

