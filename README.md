**HT embedding and detection**
===========================

**Overview**
----------

This code illustrates the proposed approach for effective and scalable HT detection in gate-level netlist. It consists of two main experimental stages:

* Automated analysis of the available reference HT-injected circuits and training of the detection ML model
* Evaluation of the trained model

This package contain the TRIT-TC benchmark in `TRIT-TC` folder (Contributed By Swarup Bhunia, University of Florida, Jonathan Cruz, jonc205@ufl.edu, University of Florida)

**Getting Started**
-----------------

This code uses the PyTorch ML library. Please install the version that fits your OS and hardware from https://pytorch.org.

This code is designed for short interactive experiments. Experiments are defined as functions in corresponding Python source files (such as `detection_framework.py`), and are called from the Python file top-level statement by executing the corresponding scripts (e.g., `python ./detection_framework.py`). To run another experiment, remove or comment out the top-level statements of existing experiments, and add a new function call instead.

Experiments may require input artifacts, which are usually loaded from the workspace files (artifact files are defined in function parameters), and may produce output artifacts, which are usually saved to the workspace files. Besides these artifacts, some experiments may output the experiment results in a human-readable format into `stdout`.

Usually, multiple experiments form a DAG pipeline, where the artifacts produced by the upstream experiments are consumed by the downstream experiments, forming a HT detection flow model. However, this pipeline is not automated, and it is the end-user's responsibility to run experiments in the desired order and to assert consistency between produced and consumed artifacts.

This structure allows users to effectively debug each individual experiment and manually compare the output artifacts with expected values.

Artifacts are not included into the package, however, they may be generated with experiments.

**Experiments overview**
------------------------

The `./detection_framework.py` file contains experiments that perform the following:

* Parsing VHDL files into a graph representation
* Graph embedding
* Defining the embedding convolutional model
* Generating training data from known HT-compromised circuits
* Model inference

The `DetectionFramework` class includes functions that:

* Parse VHDL files into a graph representation
* Extract labels from the VHDL file meta-information
* Perform graph operations (such as sampling neighbors, shortest path computations, etc.)
* Perform graph embedding into a vector cloud
* Perform operations on a vector cloud
* Map the corresponding graph elements to the embedding vectors
* Map the embedding vectors/graph elements to the corresponding training label

Experiments include:

* `test()`, which performs the embedding of a single IC design and analyzes metrics of safe components and HT components
* `generate_trainset_from_benchmarks()`, which generates a training set from the given reference VHDL files
* `test_mlp()`, which evaluates the supervised embedding convolution model on the given labeled data
* And many others!

The `./queries.py` file contains experiments that evaluate and report the performance of the trained model. Evaluation experiments include the proposed model stability analysis and hypothesis evaluation.

Interactive Jupyter Python notebooks (`*.ipynb`) consist interactive experiments, which inspired the proposed HT detection approach, but not directly connected with it. These experiments gave us a lot of impotact insights on IC data parsing, graph embedding, embeddings visualisation and infected IC strtucture. These experiments may use modified TRIC-TC files from `benches`, `graph`, and `tmp` folders, as well as `*.v` artifacts from the root workspace directory.