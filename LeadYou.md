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

* manager
	+ check the update_info names (if exist) are distinct each other

* metrics
	+ prepare many metrics (make it easy to logs ...)
	+ save etrics log

* event setting
	+ each some couples of iterations
	+ evaluation (train dataset) at each epoch
	+ evaluation (validation dataset) at each epoch
	+ optimizer.grad_zero() at epoch starts

* visualization
	+ test : visdom
	+ add : tensorwatch

* backend integration
	+ compatibility with chainer

#### In future

* inference

* deploy

