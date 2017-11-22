import cPickle
import optparse
import os
import re
import sys
import time

model_settings_pattern = re.compile(
	'\d+-\d+-hdp-I(?P<iteration>\d+)-S(?P<snapshot>\d+)-aa(?P<alpha>[\d\.]+)-ag(?P<gamma>[\d\.]+)-ae(?P<eta>[\d\.]+)(-smh(?P<smh>[\d]+))?(-sp(?P<sp>[\d]+)-mp(?P<mp>[\d]+))?')


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		input_directory=None,
		# output_directory=None,
		snapshot_index=500,
		training_iterations=1000
	)
	# parameter set 1
	parser.add_option("--input_directory", type="string", dest="input_directory",
	                  help="input directory [None]")
	# parser.add_option("--output_directory", type="string", dest="output_directory",
	# help="output directory [None]")
	parser.add_option("--snapshot_index", type="int", dest="snapshot_index",
	                  help="snapshot index [500]")

	parser.add_option("--training_iterations", type="int", dest="training_iterations",
	                  help="number of training iterations [1000]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	assert (options.input_directory != None)
	input_directory = options.input_directory
	if not os.path.exists(input_directory):
		sys.stderr.write("model directory %s not exists...\n" % (input_directory))
		return
	input_directory = input_directory.rstrip("/")
	model_settings = os.path.basename(input_directory)
	snapshot_index = options.snapshot_index

	matches = re.match(model_settings_pattern, model_settings)
	training_iterations = int(matches.group('iteration'))
	snapshot_interval = int(matches.group('snapshot'))
	if options.training_iterations != training_iterations:
		model_settings = model_settings.replace("-I%d-" % (training_iterations),
		                                        "-I%d-" % (options.training_iterations))
		output_directory = os.path.join(os.path.dirname(input_directory), model_settings)
		assert (not os.path.exists(output_directory))
		os.rename(input_directory, output_directory)
		training_iterations = options.training_iterations

		print 'successfully rename directory from %s to %s...' % (input_directory, output_directory)
	else:
		output_directory = input_directory

	hdp = cPickle.load(open(os.path.join(output_directory, "model-%d" % snapshot_index), "rb"))
	print 'successfully load model snpashot %s...' % (os.path.join(output_directory, "model-%d" % snapshot_index))

	for iteration in xrange(hdp._iteration_counter, training_iterations):
		clock = time.time()
		log_likelihood = hdp.learning()
		clock = time.time() - clock
		print 'training iteration %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (
			hdp._iteration_counter, clock, hdp._K, log_likelihood)

		# Save lambda, the parameters to the variational distributions over topics, and batch_gamma, the parameters to the variational distributions over topic weights for the articles analyzed in the last iteration.
		if (hdp._iteration_counter % snapshot_interval == 0):
			hdp.export_beta(os.path.join(output_directory, 'exp_beta-' + str(hdp._iteration_counter)), 50)
			# numpy.savetxt(os.path.join(output_directory, 'n_kv-' + str(hdp._iteration_counter)), hdp._n_kv, fmt="%d")
			model_snapshot_path = os.path.join(output_directory, 'model-' + str(hdp._iteration_counter))
			cPickle.dump(hdp, open(model_snapshot_path, 'wb'))

		# gamma_path = os.path.join(output_directory, 'gamma.txt')
		# numpy.savetxt(gamma_path, hdp._document_topic_distribution)

		# topic_inactive_counts_path = os.path.join(output_directory, "topic_inactive_counts.txt")
		# numpy.savetxt(topic_inactive_counts_path, hdp._topic_inactive_counts)


if __name__ == '__main__':
	main()
