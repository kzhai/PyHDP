PyHDP
==========

PyHDP is a Hierarchical Dirichlet Process package.

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyHDP).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyHDP package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e.,

	$PROJECT_SPACE/src/PyHDP

To prepare the example dataset,

	tar zxvf pnas-abstract.tar.gz

To launch PyHDP, first redirect to the directory of PyHDP source code,

	cd $PROJECT_SPACE/src/PyHDP

and run the following command on example dataset,

	python -m launch_train --input_directory=./pnas-abstract --output_directory=./ --training_iterations=50

The generic argument to run PyHDP is

	python -m launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --training_iterations=$NUMBER_OF_ITERATIONS

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any circumstances, you may also get help information and usage hints by running the following command

	python -m launch_train --help