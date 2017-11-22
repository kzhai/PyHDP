import cPickle
import datetime
import optparse
import os
import time


def parse_args():
	parser = optparse.OptionParser()
	parser.set_defaults(  # parameter set 1
		input_directory=None,
		output_directory=None,
		# corpus_name=None,

		# parameter set 2
		alpha_eta=-1,
		alpha_alpha=0.1,
		alpha_gamma=0.1,

		# parameter set 3
		training_iterations=1000,
		snapshot_interval=100,
		# resample_topics=False,
		# hash_oov_words=False,

		# parameter set 4
		split_proposal=0,
		merge_proposal=0,
		split_merge_heuristics=-1,
	)
	# parameter set 1
	parser.add_option("--input_directory", type="string", dest="input_directory",
	                  help="input directory [None]")
	parser.add_option("--output_directory", type="string", dest="output_directory",
	                  help="output directory [None]")
	# parser.add_option("--corpus_name", type="string", dest="corpus_name",
	# help="the corpus name [None]")

	# parameter set 2
	parser.add_option("--alpha_eta", type="float", dest="alpha_eta",
	                  help="hyper-parameter for Dirichlet distribution of vocabulary [1.0/number_of_types]")
	parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
	                  help="hyper-parameter for top level Dirichlet process of distribution over topics [0.1]")
	parser.add_option("--alpha_gamma", type="float", dest="alpha_gamma",
	                  help="hyper-parameter for bottom level Dirichlet process of distribution over topics [0.1]")

	# parameter set 3
	parser.add_option("--training_iterations", type="int", dest="training_iterations",
	                  help="number of training iterations [1000]")
	parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
	                  help="snapshot interval [100]")
	# parser.add_option("--resample_topics", action="store_true", dest="resample_topics",
	# help="resample topics [False]")
	# parser.add_option("--hash_oov_words", action="store_true", dest="hash_oov_words",
	# help="hash out-of-vocabulary words to run this model in pseudo infinite vocabulary mode [False]")

	# parameter set 4
	parser.add_option("--merge_proposal", type="int", dest="merge_proposal",
	                  help="propose merge operation via [ " +
	                       "0 (default): metropolis-hastings, " +
	                       "1: restricted gibbs sampler and metropolis-hastings, " +
	                       "2: gibbs sampler and metropolis-hastings " +
	                       "]")
	parser.add_option("--split_proposal", type="int", dest="split_proposal",
	                  help="propose split operation via [ " +
	                       "0 (default): metropolis-hastings, " +
	                       "1: restricted gibbs sampler and metropolis-hastings, " +
	                       "2: sequential allocation and metropolis-hastings " +
	                       "]")
	parser.add_option("--split_merge_heuristics", type="int", dest="split_merge_heuristics",
	                  help="split-merge heuristics [ " +
	                       "-1 (default): no split-merge operation, " +
	                       "0: component resampling, " +
	                       "1: random choose candidate clusters by points, " +
	                       "2: random choose candidate clusters by point-cluster, " +
	                       "3: random choose candidate clusters by clusters " +
	                       "]")

	(options, args) = parser.parse_args()
	return options


def main():
	options = parse_args()

	# parameter set 1
	# assert(options.corpus_name!=None)
	assert (options.input_directory != None)
	assert (options.output_directory != None)

	input_directory = options.input_directory
	input_directory = input_directory.rstrip("/")
	corpus_name = os.path.basename(input_directory)

	output_directory = options.output_directory
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
	output_directory = os.path.join(output_directory, corpus_name)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory)

	# Document
	train_docs = []
	input_doc_stream = open(os.path.join(input_directory, 'doc.dat'), 'r')
	for line in input_doc_stream:
		train_docs.append(line.strip().lower())
	print "successfully load all training documents..."

	# Vocabulary
	dictionary_file = os.path.join(input_directory, 'voc.dat')
	input_voc_stream = open(dictionary_file, 'r')
	vocab = []
	for line in input_voc_stream:
		vocab.append(line.strip().lower().split()[0])
	vocab = list(set(vocab))
	print "successfully load all the words from %s..." % (dictionary_file)

	# parameter set 2
	alpha_eta = 1.0 / len(vocab)
	if options.alpha_eta > 0:
		alpha_eta = options.alpha_eta
	assert (options.alpha_alpha > 0)
	alpha_alpha = options.alpha_alpha
	assert (options.alpha_gamma > 0)
	alpha_gamma = options.alpha_gamma

	# parameter set 3
	if options.training_iterations > 0:
		training_iterations = options.training_iterations
	if options.snapshot_interval > 0:
		snapshot_interval = options.snapshot_interval

	# resample_topics = options.resample_topics
	# hash_oov_words = options.hash_oov_words

	# parameter set 4
	split_merge_heuristics = options.split_merge_heuristics
	split_proposal = options.split_proposal
	merge_proposal = options.merge_proposal

	# create output directory
	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S") + ""
	suffix += "-%s" % ("hdp")
	suffix += "-I%d" % (training_iterations)
	suffix += "-S%d" % (snapshot_interval)
	suffix += "-aa%f" % (alpha_alpha)
	suffix += "-ag%f" % (alpha_gamma)
	suffix += "-ae%f" % (alpha_eta)
	# suffix += "-%s" % (resample_topics)
	# suffix += "-%s" % (hash_oov_words)
	if split_merge_heuristics >= 0:
		suffix += "-smh%d" % (split_merge_heuristics)
	if split_merge_heuristics >= 1:
		suffix += "-sp%d" % (split_proposal)
		suffix += "-mp%d" % (merge_proposal)
	suffix += "/"

	output_directory = os.path.join(output_directory, suffix)
	os.mkdir(os.path.abspath(output_directory))

	# store all the options to a input_doc_stream
	options_output_file = open(output_directory + "option.txt", 'w')
	# parameter set 1
	options_output_file.write("input_directory=" + input_directory + "\n")
	options_output_file.write("corpus_name=" + corpus_name + "\n")
	options_output_file.write("dictionary_file=" + str(dictionary_file) + "\n")
	# parameter set 2
	options_output_file.write("alpha_eta=" + str(alpha_eta) + "\n")
	options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n")
	options_output_file.write("alpha_gamma=" + str(alpha_gamma) + "\n")
	# parameter set 3
	options_output_file.write("training_iteration=%d\n" % training_iterations)
	options_output_file.write("snapshot_interval=%d\n" % snapshot_interval)
	# options_output_file.write("resample_topics=%s\n" % resample_topics)
	# options_output_file.write("hash_oov_words=%s\n" % hash_oov_words)
	# parameter set 4
	if split_merge_heuristics >= 0:
		options_output_file.write("split_merge_heuristics=%d\n" % split_merge_heuristics)
	if split_merge_heuristics >= 1:
		options_output_file.write("split_proposal=%d\n" % split_proposal)
		options_output_file.write("merge_proposal=%d\n" % merge_proposal)
	options_output_file.close()

	print "========== ========== ========== ========== =========="
	# parameter set 1
	print "output_directory=" + output_directory
	print "input_directory=" + input_directory
	print "corpus_name=" + corpus_name
	print "dictionary_file=" + str(dictionary_file)
	# parameter set 2
	print "alpha_eta=" + str(alpha_eta)
	print "alpha_alpha=" + str(alpha_alpha)
	print "alpha_gamma=" + str(alpha_gamma)
	# parameter set 3
	print "training_iteration=%d" % (training_iterations)
	print "snapshot_interval=%d" % (snapshot_interval)
	# print "resample_topics=%s" % (resample_topics)
	# print "hash_oov_words=%s" % (hash_oov_words)
	# parameter set 4
	if split_merge_heuristics >= 0:
		print "split_merge_heuristics=%d" % (split_merge_heuristics)
	if split_merge_heuristics >= 1:
		print "split_proposal=%d" % split_proposal
		print "merge_proposal=%d" % merge_proposal
	print "========== ========== ========== ========== =========="

	import monte_carlo
	hdp = monte_carlo.MonteCarlo(split_merge_heuristics, split_proposal, merge_proposal)
	hdp._initialize(train_docs, vocab, alpha_alpha, alpha_gamma, alpha_eta)

	hdp.export_beta(os.path.join(output_directory, 'exp_beta-' + str(hdp._iteration_counter)), 50)
	# numpy.savetxt(os.path.join(output_directory, 'n_kv-' + str(hdp._iteration_counter)), hdp._n_kv, fmt="%d")

	for iteration in xrange(training_iterations):
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
