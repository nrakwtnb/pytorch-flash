## config

* device
	+ name : str `cpu` ...
	+ gpu : int?
	+ num_gpu : int

* train
	+ epochs
	+ batch_size
	+ train_batch_size
		- batch_sizeと統一する


* handler
	+ early_stopping
		+ patience
		+ score_function : optional
		+ score_name : optional

* others

* objects
	+ data
		+ train_loader
		+ val_loader
	* update_info_list
	* evaluate_info_list
	* metrics
	* metrics_log
	* vis_tool

metrics_logとvis_toolをまとめる？

## update info

* name
* skip_condition : optional
* break_condition : optional
* pre_condition
* post_condition
* model
* optimizer : optional
* loss_fn
* num_iter : optional, default=1

all necessary

