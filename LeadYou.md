### ToDo

* model
	+ allow model to tell what data it needs
	+ Ex.
		- > my_model.show_input_info
		- "images" : FloatTensor, size=(*,3,224,224)
		- "z_hidden" : FloatTensor, size=(*,10)
	+ input wrapper
		- See `dev.py`


* event setting
	+ each some couples of iterations
	+ evaluation (train dataset) at each epoch
	+ evaluation (validation dataset) at each epoch


