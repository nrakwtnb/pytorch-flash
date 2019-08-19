
# pytorch-flash

## Goal

This is an experimental project to realize:

* fast implementations and experiments on Deep Learning projects
* simple codes to understand and maintain with ease
	* unified DL interfaces as possible

* less effort and complex, more productive and flexible

## Description

* deep learning implementations wrapper based on pytorch and pytorch-ignite
* unofficial private project ( unrelated to the pytorch developer teams )

Caution:

This is experimental and there occur unexpected behaviour. Please notice me them !

## How To

To appear ...

## ToDo

* interface (API)
	* []: simpler training description around update additions
	* []: events definition
	* []: simplify transformer attachments through dataloader -> model -> loss

* train manager
	* []: refactoring metrics setup (separate metrics manager from train manager)
	* []: seed management/controller (ensure reproductions)
	* []: gpu management (including multi GPUs)
	* []: save, load and resume
	* []: update visualization
	* []: test-run (or dry-run)

* engine
	* []: provide some buffer in ignite.State class (e.g. replay buffer)

## Schedule

* v0.0 alpha : November, 2019
	+ add several functions
	+ unify iterfaces as possible
	+ several tests and debugs
	+ make examples

## Future Works

* automatic flow generation

* efficient debug functions
	+ based on tensorwatch ?

* parallel compuations
	+ data parallel
	+ multi GPUs

* other DL framework backends, in particular, chainer

* connection to model deployments

