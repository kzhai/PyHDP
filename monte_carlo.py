"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by Chong Wang (chongw@cs.princeton.edu).
Implements uncollapsed Gibbs sampling for the hierarchical Dirichlet process (HDP).

References:
[1] Chong Wang and David Blei, A Split-Merge MCMC Algorithm for the Hierarchical Dirichlet Process, available online www.cs.princeton.edu/~chongw/papers/sm-hdp.pdf.
"""

import copy
import random
import sys

import nltk
import numpy
import scipy
import scipy.special
import scipy.stats

negative_infinity = -1e500

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')


class MonteCarlo(object):
	"""
	@param truncation_level: the maximum number of clusters, used for speeding up the computation
	@param snapshot_interval: the interval for exporting a snapshot of the model
	"""

	def __init__(self,
	             split_merge_heuristics=-1,
	             split_proposal=0,
	             merge_proposal=0,
	             split_merge_iteration=1,
	             restrict_gibbs_sampling_iteration=10,
	             component_resampling_interval=1,
	             hyper_optimizing_interval=10,
	             hash_oov_words=False
	             ):
		self._split_merge_heuristics = split_merge_heuristics
		self._split_proposal = split_proposal
		self._merge_proposal = merge_proposal

		self._split_merge_iteration = split_merge_iteration
		self._restrict_gibbs_sampling_iteration = restrict_gibbs_sampling_iteration

		self._component_resampling_interval = component_resampling_interval
		self._hyper_optimizing_interval = hyper_optimizing_interval

		self._hash_oov_words = hash_oov_words

	"""
	@param data: a N-by-D numpy array object, defines N points of D dimension
	@param K: number of topics, number of broke sticks
	@param alpha: the probability of f_{k_{\mathsf{new}}}^{-x_{dv}}(x_{dv}), the prior probability density for x_{dv}
	@param gamma: the smoothing value for a table to be assigned to a new topic
	@param eta: the smoothing value for a word to be assigned to a new topic
	"""

	def _initialize(self,
	                corpus,
	                vocab,
	                alpha_alpha,
	                alpha_gamma,
	                alpha_eta=0
	                ):
		self._iteration_counter = 0

		self._word_to_index = {}
		self._index_to_word = {}
		for word in set(vocab):
			self._index_to_word[len(self._index_to_word)] = word
			self._word_to_index[word] = len(self._word_to_index)

		self._vocabulary = self._word_to_index.keys()
		self._vocabulary_size = len(self._vocabulary)

		# top level smoothing
		self._alpha_alpha = alpha_alpha
		# bottom level smoothing
		self._alpha_gamma = alpha_gamma
		# vocabulary smoothing
		if alpha_eta <= 0:
			self._alpha_eta = 1.0 / self._vocabulary_size
		else:
			self._alpha_eta = alpha_eta

		# initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
		self._corpus = self.parse_doc_list(corpus)

		# initialize the size of the collection, i.e., total number of documents.
		self._D = len(self._corpus)

		model_parameter = self.random_initialization(1, 1)
		# model_parameter = self.random_initialization(30, 1)
		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		self._K = proposed_K

		self._n_kv = proposed_n_kv
		self._m_k = proposed_m_k
		self._n_dk = proposed_n_dk

		self._n_dt = proposed_n_dt

		self._t_dv = proposed_t_dv
		self._k_dt = proposed_k_dt

	# print "accumulated number of tables:", self._m_k
	# print "accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T

	def random_initialization(self, number_of_topics=10, number_of_tables=5):
		proposed_K = number_of_topics

		# initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
		proposed_n_kv = numpy.zeros((proposed_K, self._vocabulary_size))

		# initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
		proposed_m_k = numpy.zeros(proposed_K)

		# initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
		proposed_n_dk = numpy.zeros((self._D, proposed_K))

		# random initialize all documents

		# initialize the table information vectors indexed by document id and word id, i.e., t{j i}
		proposed_t_dv = {}
		# initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
		proposed_k_dt = {}
		# initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
		proposed_n_dt = {}

		# we assume all words in a document belong to one table which was assigned to topic 0
		for document_index in xrange(self._D):
			# initialize the table information vector indexed by document and records down which table a word belongs to
			proposed_t_dv[document_index] = numpy.random.randint(0, number_of_tables,
			                                                     len(self._corpus[document_index]))

			# self._k_dt records down which topic a table was assigned to
			proposed_k_dt[document_index] = numpy.random.randint(0, number_of_topics, number_of_tables)
			# assert (len(proposed_k_dt[document_index]) == len(numpy.unique(proposed_t_dv[document_index]))), (len(proposed_k_dt[document_index]), proposed_t_dv[document_index], len(numpy.unique(proposed_t_dv[document_index])))

			# word_count_table records down the number of words sit on every table
			# proposed_n_dt[document_index] = numpy.zeros(number_of_tables, dtype=numpy.int) + len(self._corpus[document_index])
			proposed_n_dt[document_index] = numpy.zeros(number_of_tables, dtype=numpy.int)
			for word_pos in xrange(len(self._corpus[document_index])):
				table_index = proposed_t_dv[document_index][word_pos]
				proposed_n_dt[document_index][table_index] += 1

				word_index = self._corpus[document_index][word_pos]
				topic_index = proposed_k_dt[document_index][table_index]
				proposed_n_kv[topic_index, word_index] += 1
				proposed_n_dk[document_index, topic_index] += 1

			# assert (len(proposed_n_dt[document_index]) == len(numpy.unique(proposed_t_dv[document_index]))), (len(proposed_n_dt[document_index]), len(numpy.unique(proposed_t_dv[document_index])))
			assert (numpy.sum(proposed_n_dt[document_index]) == len(self._corpus[document_index]))

			for table_index in xrange(len(proposed_k_dt[document_index])):
				if proposed_n_dt[document_index][table_index] == 0:
					continue
				topic_index = proposed_k_dt[document_index][table_index]
				proposed_m_k[topic_index] += 1

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		model_parameter = self.compact_params(model_parameter)
		# self.model_assertion(model_parameter)
		return model_parameter

	def parse_doc_list(self, docs):
		if (type(docs).__name__ == 'str'):
			temp = list()
			temp.append(docs)
			docs = temp

		D = len(docs)
		wordids = list()

		for d in xrange(D):
			words = docs[d].split()
			wordid = list()
			for word in words:
				if (word in self._vocabulary):
					wordtoken = self._word_to_index[word]
					wordid.append(wordtoken)
				else:
					if self._hash_oov_words:
						wordtoken = hash(word) % len(self._vocabulary)
						wordid.append(wordtoken)

			wordids.append(wordid)

		return wordids

	def learning(self):
		self._iteration_counter += 1

		self.sample_cgs()

		if self._split_merge_heuristics == 0:
			if self._iteration_counter % self._component_resampling_interval == 0:
				self.resample_components()
		elif self._split_merge_heuristics > 0:
			self.split_merge()

		if self._hyper_optimizing_interval > 0 and self._iteration_counter % self._hyper_optimizing_interval == 0:
			# self.optimize_log_hyperparameters()
			self.optimize_hyperparameters()

		# print "accumulated number of tables:", self._m_k
		# print "accumulated number of tokens:", numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis].T

		return self.log_posterior()

	"""
	sample the data to train the parameters
	"""

	def sample_cgs(self):
		# self.model_assertion()

		# sample the total data
		# for document_index in numpy.random.permutation(xrange(self._D)):
		for document_index in xrange(self._D):
			# sample word assignment, see which table it should belong to
			# for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
			for word_index in xrange(len(self._corpus[document_index])):
				# get the word_id of at the word_index of the document_index
				word_id = self._corpus[document_index][word_index]

				# retrieve the old_table_id of the current word of current document
				old_table_id = self._t_dv[document_index][word_index]
				# retrieve the old_topic_id of the table that current word of current document sit on
				old_topic_id = self._k_dt[document_index][old_table_id]

				self._n_dt[document_index][old_table_id] -= 1
				assert (numpy.all(self._n_dt[document_index] >= 0))
				self._n_kv[old_topic_id, word_id] -= 1
				assert (numpy.all(self._n_kv >= 0))
				self._n_dk[document_index, old_topic_id] -= 1
				assert (numpy.all(self._n_dk >= 0))

				# if current table in current document becomes empty
				if self._n_dt[document_index][old_table_id] == 0:
					# adjust the table counts
					self._m_k[old_topic_id] -= 1
					assert (numpy.all(self._m_k >= 0))

				assert (numpy.all(self._k_dt[document_index] >= 0))

				n_k = numpy.sum(self._n_kv, axis=1)
				assert (len(n_k) == self._K)
				f = (self._n_kv[:, word_id] + self._alpha_eta) / (n_k + self._vocabulary_size * self._alpha_eta)
				f_new = self._alpha_alpha / self._vocabulary_size
				f_new += numpy.sum(self._m_k * f)
				f_new /= (numpy.sum(self._m_k) + self._alpha_alpha)

				# compute the probability of this word sitting at every table
				table_probability = f[self._k_dt[document_index]] * self._n_dt[document_index]
				table_probability = numpy.hstack((table_probability, numpy.zeros(1)))
				# compute the probability of current word sitting on a new table, the prior probability is self._alpha_gamma
				table_probability[len(self._k_dt[document_index])] = self._alpha_gamma * f_new

				# sample a new table this word should sit in
				table_probability /= numpy.sum(table_probability)
				cdf = numpy.cumsum(table_probability)
				new_table_id = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])

				# if current word sits on a new table, we need to get the topic of that table
				if new_table_id == len(self._k_dt[document_index]):
					if self._n_dt[document_index][old_table_id] == 0:
						# if the old table is empty, reuse it
						new_table_id = old_table_id
					else:
						# else expand the vectors to fit in new table
						self._n_dt[document_index] = numpy.hstack(
							(self._n_dt[document_index], numpy.zeros(1, dtype=numpy.int)))
						self._k_dt[document_index] = numpy.hstack(
							(self._k_dt[document_index], numpy.zeros(1, dtype=numpy.int)))

					assert (len(self._n_dt) == self._D and numpy.all(self._n_dt[document_index] >= 0))
					assert (len(self._k_dt) == self._D and numpy.all(self._k_dt[document_index] >= 0))
					assert (len(self._n_dt[document_index]) == len(self._k_dt[document_index]))

					# compute the probability of this table having every topic
					topic_probability = numpy.zeros(self._K + 1)
					topic_probability[:self._K] = self._m_k * f
					topic_probability[self._K] = self._alpha_alpha / self._vocabulary_size

					# sample a new topic this table should be assigned
					topic_probability /= numpy.sum(topic_probability)
					cdf = numpy.cumsum(topic_probability)
					new_topic_id = numpy.int(numpy.nonzero(cdf >= numpy.random.random())[0][0])

					# if current table requires a new topic
					if new_topic_id == self._K:
						if self._m_k[old_topic_id] == 0:
							# if the old topic is empty, reuse it
							new_topic_id = old_topic_id
						else:
							# else expand matrices to fit in new topic
							self._K += 1
							self._n_kv = numpy.vstack(
								(self._n_kv, numpy.zeros((1, self._vocabulary_size), dtype=numpy.int)))
							assert (self._n_kv.shape == (self._K, self._vocabulary_size))
							self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1), dtype=numpy.int)))
							assert (self._n_dk.shape == (self._D, self._K))
							self._m_k = numpy.hstack((self._m_k, numpy.zeros(1, dtype=numpy.int)))
							assert (len(self._m_k) == self._K)

					# assign current table to new topic
					self._k_dt[document_index][new_table_id] = new_topic_id

				# assign current word to new table
				self._t_dv[document_index][word_index] = new_table_id

				# retrieve the new_table_id of the current word of current document
				new_table_id = self._t_dv[document_index][word_index]
				# retrieve the new_topic_id of the table that current word of current document sit on
				new_topic_id = self._k_dt[document_index][new_table_id]

				self._n_dt[document_index][new_table_id] += 1
				assert (numpy.all(self._n_dt[document_index] >= 0))
				self._n_kv[new_topic_id, word_id] += 1
				assert (numpy.all(self._n_kv >= 0))
				self._n_dk[document_index, new_topic_id] += 1
				assert (numpy.all(self._n_dk >= 0))

				# if a new table is created in current document
				if self._n_dt[document_index][new_table_id] == 1:
					# adjust the table counts
					self._m_k[new_topic_id] += 1

				assert (numpy.all(self._m_k >= 0))
				assert (numpy.all(self._k_dt[document_index] >= 0))

			# sample table assignment, see which topic it should belong to
			# for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
			for table_index in xrange(len(self._k_dt[document_index])):
				self.sample_tables(document_index, table_index)

		# compact all the parameters, including removing unused topics and unused tables
		self.compact_params()

	# self.model_assertion()

	def sample_tables(self, document_index, table_index):
		# if this table is empty, skip the sampling directly
		if self._n_dt[document_index][table_index] <= 0:
			return

		old_topic_id = self._k_dt[document_index][table_index]

		# find the index of the words sitting on the current table
		selected_word_index = numpy.nonzero(self._t_dv[document_index] == table_index)[0]
		# find the frequency distribution of the words sitting on the current table
		selected_word_freq_dist = nltk.probability.FreqDist(
			[self._corpus[document_index][term] for term in list(selected_word_index)])
		assert (self._n_dt[document_index][table_index] == selected_word_freq_dist.N())

		# adjust the statistics of all model parameter
		self._m_k[old_topic_id] -= 1
		assert numpy.all(self._m_k >= 0)
		self._n_dk[document_index, old_topic_id] -= self._n_dt[document_index][table_index]
		assert numpy.all(self._n_dk >= 0)
		for word_id in selected_word_freq_dist:
			self._n_kv[old_topic_id, word_id] -= selected_word_freq_dist[word_id]
			assert (self._n_kv[old_topic_id, word_id] >= 0)

		# compute the probability of assigning current table every topic
		topic_log_probability = numpy.zeros(self._K + 1)

		topic_log_probability[self._K] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
		topic_log_probability[self._K] -= scipy.special.gammaln(
			self._n_dt[document_index][table_index] + self._vocabulary_size * self._alpha_eta)
		for word_id in selected_word_freq_dist:
			topic_log_probability[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._alpha_eta)
			topic_log_probability[self._K] -= scipy.special.gammaln(self._alpha_eta)
		# topic_log_probability[self._K] -= selected_word_freq_dist.N() * scipy.special.gammaln(self._alpha_eta)
		topic_log_probability[self._K] += numpy.log(self._alpha_alpha)

		n_k = numpy.sum(self._n_kv, axis=1)
		assert (len(n_k) == (self._K))

		topic_log_probability[:self._K] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta + n_k)
		topic_log_probability[:self._K] -= scipy.special.gammaln(
			self._vocabulary_size * self._alpha_eta + n_k + self._n_dt[document_index][table_index])
		for word_id in selected_word_freq_dist:
			topic_log_probability[:self._K] += scipy.special.gammaln(
				selected_word_freq_dist[word_id] + self._n_kv[:, word_id] + self._alpha_eta)
			topic_log_probability[:self._K] -= scipy.special.gammaln(self._n_kv[:, word_id] + self._alpha_eta)
		# compute the prior if we move this table from this topic
		topic_log_probability[:self._K] += numpy.log(self._m_k)

		# normalize the distribution and sample new topic assignment for this topic
		# topic_log_probability = numpy.exp(topic_log_probability)
		# topic_log_probability = topic_log_probability/numpy.sum(topic_log_probability)
		# topic_log_probability = numpy.exp(log_normalize(topic_log_probability))
		topic_log_probability -= scipy.misc.logsumexp(topic_log_probability)
		topic_probability = numpy.exp(topic_log_probability)

		cdf = numpy.cumsum(topic_probability)
		new_topic_id = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])

		# if the table is assigned to a new topic
		if new_topic_id == self._K:
			if self._m_k[old_topic_id] <= 0:
				# if old topic is empty, reuse it
				new_topic_id = old_topic_id
			else:
				# else expand all matrices to fit new topics
				self._K += 1
				self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._vocabulary_size), dtype=numpy.int)))
				assert (self._n_kv.shape == (self._K, self._vocabulary_size))
				self._n_dk = numpy.hstack((self._n_dk, numpy.zeros((self._D, 1), dtype=numpy.int)))
				assert (self._n_dk.shape == (self._D, self._K))
				self._m_k = numpy.hstack((self._m_k, numpy.zeros(1, dtype=numpy.int)))
				assert (len(self._m_k) == self._K)

		# assign this table to new topic
		self._k_dt[document_index][table_index] = new_topic_id

		# adjust the statistics of all model parameter
		self._m_k[new_topic_id] += 1
		self._n_dk[document_index, new_topic_id] += self._n_dt[document_index][table_index]
		for word_id in selected_word_freq_dist:
			self._n_kv[new_topic_id, word_id] += selected_word_freq_dist[word_id]

	def optimize_log_hyperparameters(self, hyperparameter_samples=10, hyperparameter_step_size=1.0,
	                                 hyperparameter_maximum_iteration=10):
		old_hyper_parameters = [self._alpha_alpha, self._alpha_gamma, self._alpha_eta]
		old_hyper_parameters = numpy.asarray(old_hyper_parameters)
		old_log_hyper_parameters = numpy.log(old_hyper_parameters)

		for ii in xrange(hyperparameter_samples):
			log_likelihood_old = self.log_posterior()
			log_likelihood_new = numpy.log(numpy.random.random()) + log_likelihood_old

			l = old_log_hyper_parameters - numpy.random.random(
				len(old_log_hyper_parameters)) * hyperparameter_step_size
			r = old_log_hyper_parameters + hyperparameter_step_size

			for jj in xrange(hyperparameter_maximum_iteration):
				new_log_hyper_parameters = l + numpy.random.random(len(old_log_hyper_parameters)) * (r - l)
				lp_test = self.log_posterior(None, numpy.exp(new_log_hyper_parameters))

				if lp_test > log_likelihood_new:
					self._alpha_alpha = numpy.exp(new_log_hyper_parameters[0])
					self._alpha_gamma = numpy.exp(new_log_hyper_parameters[1])
					self._alpha_eta = numpy.exp(new_log_hyper_parameters[2])
					old_log_hyper_parameters = new_log_hyper_parameters
					# print "update hyperparameter to %s" % (numpy.exp(new_log_hyper_parameters))
					break
				else:
					for dd in xrange(len(new_log_hyper_parameters)):
						if new_log_hyper_parameters[dd] < old_log_hyper_parameters[dd]:
							l[dd] = new_log_hyper_parameters[dd]
						else:
							r[dd] = new_log_hyper_parameters[dd]
						assert l[dd] <= old_log_hyper_parameters[dd]
						assert r[dd] >= old_log_hyper_parameters[dd]

	def optimize_hyperparameters(self, hyperparameter_samples=10, hyperparameter_step_size=1.0,
	                             hyperparameter_maximum_iteration=10):
		old_hyper_parameters = [self._alpha_alpha, self._alpha_gamma, self._alpha_eta]
		old_hyper_parameters = numpy.asarray(old_hyper_parameters)
		# old_hyper_parameters = numpy.log(old_hyper_parameters)

		for ii in xrange(hyperparameter_samples):
			step_size = old_hyper_parameters
			step_size[numpy.nonzero(step_size > hyperparameter_step_size)[0]] = hyperparameter_step_size

			log_likelihood_old = self.log_posterior()
			log_likelihood_new = numpy.log(numpy.random.random()) + log_likelihood_old

			l = old_hyper_parameters - numpy.random.random(len(old_hyper_parameters)) * step_size
			r = old_hyper_parameters + step_size

			for jj in xrange(hyperparameter_maximum_iteration):
				new_hyper_parameters = l + numpy.random.random(len(old_hyper_parameters)) * (r - l)
				lp_test = self.log_posterior(None, new_hyper_parameters)

				if lp_test > log_likelihood_new:
					self._alpha_alpha = new_hyper_parameters[0]
					self._alpha_gamma = new_hyper_parameters[1]
					self._alpha_eta = new_hyper_parameters[2]
					old_hyper_parameters = new_hyper_parameters
					# print "update hyperparameter to %s" % (new_hyper_parameters)
					break
				else:
					for dd in xrange(len(new_hyper_parameters)):
						if new_hyper_parameters[dd] < old_hyper_parameters[dd]:
							l[dd] = new_hyper_parameters[dd]
						else:
							r[dd] = new_hyper_parameters[dd]
						assert l[dd] <= old_hyper_parameters[dd]
						assert r[dd] >= old_hyper_parameters[dd]

	def split_merge(self):
		for iteration in xrange(self._split_merge_iteration):
			topic_probability = 1.0 * self._m_k / numpy.sum(self._m_k)

			if self._split_merge_heuristics == 1:
				temp_cluster_probability = numpy.random.multinomial(1, topic_probability)[numpy.newaxis, :]
				random_label_1 = numpy.nonzero(temp_cluster_probability == 1)[1][0]
				temp_cluster_probability = numpy.random.multinomial(1, topic_probability)[numpy.newaxis, :]
				random_label_2 = numpy.nonzero(temp_cluster_probability == 1)[1][0]
			elif self._split_merge_heuristics == 2:
				temp_cluster_probability = numpy.random.multinomial(1, topic_probability)[numpy.newaxis, :]
				random_label_1 = numpy.nonzero(temp_cluster_probability == 1)[1][0]
				random_label_2 = numpy.random.randint(self._K)
			elif self._split_merge_heuristics == 3:
				random_label_1 = numpy.random.randint(self._K)
				random_label_2 = numpy.random.randint(self._K)
			else:
				sys.stderr.write("error: unrecognized split-merge heuristics %d...\n" % (self._split_merge_heuristics))
				return

			# self.model_assertion()
			if random_label_1 == random_label_2:
				self.split_metropolis_hastings(random_label_1)
			else:
				self.merge_metropolis_hastings(random_label_1, random_label_2)
			# self.model_assertion()

	def split_metropolis_hastings(self, cluster_label):
		# record down the old cluster assignment
		old_log_posterior = self.log_posterior()

		proposed_K = self._K

		proposed_n_kv = numpy.copy(self._n_kv)
		proposed_m_k = numpy.copy(self._m_k)
		proposed_n_dk = numpy.copy(self._n_dk)

		proposed_n_dt = copy.deepcopy(self._n_dt)

		proposed_t_dv = copy.deepcopy(self._t_dv)
		proposed_k_dt = copy.deepcopy(self._k_dt)

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		# self.model_assertion(model_parameter)

		if self._split_proposal == 0:
			# perform random split for split proposal
			model_parameter = self.random_split(cluster_label, model_parameter)
			if model_parameter == None:
				return
			# self.model_assertion(model_parameter)

			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter
			log_proposal_probability = (proposed_m_k[cluster_label] + proposed_m_k[proposed_K - 1] - 2) * numpy.log(2)
		elif self._split_proposal == 1:
			# perform restricted gibbs sampling for split proposal
			model_parameter = self.random_split(cluster_label, model_parameter)
			# split a singleton cluster
			if model_parameter == None:
				return
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

			# self.model_assertion(model_parameter)
			model_parameter, transition_log_probability = self.restrict_gibbs_sampling(cluster_label, proposed_K - 1,
			                                                                           model_parameter,
			                                                                           self._restrict_gibbs_sampling_iteration + 1)
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter
			# self.model_assertion(model_parameter)

			if proposed_m_k[cluster_label] == 0 or proposed_m_k[proposed_K - 1] == 0:
				return

			log_proposal_probability = transition_log_probability
		elif self._split_proposal == 2:
			# perform sequential allocation gibbs sampling for split proposal
			model_parameter = self.sequential_allocation_split(cluster_label, model_parameter)
			if model_parameter == None:
				return
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

			# self.model_assertion(model_parameter)
			log_proposal_probability = (proposed_m_k[cluster_label] + proposed_m_k[proposed_K - 1] - 2) * numpy.log(2)
		else:
			sys.stderr.write("error: unrecognized split proposal strategy %d...\n" % (self._split_proposal))

		new_log_posterior = self.log_posterior(model_parameter)

		acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior
		acceptance_probability = numpy.exp(acceptance_log_probability)

		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		if numpy.random.random() < acceptance_probability:
			print "split operation granted from %s to %s with acceptance probability %s" % (
				self._m_k, proposed_m_k, acceptance_probability)

			self._K = proposed_K

			self._n_kv = proposed_n_kv
			self._m_k = proposed_m_k
			self._n_dk = proposed_n_dk

			self._n_dt = proposed_n_dt

			self._t_dv = proposed_t_dv
			self._k_dt = proposed_k_dt

		# self.model_assertion()

	def random_split(self, component_index, model_parameter):
		# self.model_assertion(model_parameter)

		# sample the data points set
		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		if proposed_m_k[component_index] <= 1:
			return None

		number_of_unvisited_target_tables = proposed_m_k[component_index]

		proposed_K += 1

		proposed_n_dk = numpy.hstack((proposed_n_dk, numpy.zeros((self._D, 1), dtype=numpy.int)))
		assert (proposed_n_dk.shape == (self._D, proposed_K))

		proposed_n_kv = numpy.vstack((proposed_n_kv, numpy.zeros((1, self._vocabulary_size), dtype=numpy.int)))
		assert (proposed_n_kv.shape == (proposed_K, self._vocabulary_size))

		proposed_m_k = numpy.hstack((proposed_m_k, numpy.zeros(1, dtype=numpy.int)))
		assert (len(proposed_m_k) == proposed_K)

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		# self.model_assertion(model_parameter)

		# for document_index in numpy.random.permutation(xrange(self._D)):
		for document_index in xrange(self._D):
			# for table_index in numpy.random.permutation(xrange(len(proposed_k_dt[document_index]))):
			for table_index in xrange(len(proposed_k_dt[document_index])):
				if proposed_k_dt[document_index][table_index] != component_index:
					continue

				'''
				test_n_kv = numpy.zeros((proposed_K, self._vocabulary_size))
				
				for test_document_index in xrange(self._D):
					test_n_dt = [0] * len(proposed_k_dt[test_document_index])
					test_n_dk = numpy.zeros(proposed_K)
					
					# sample word assignment, see which table it should belong to
					for test_word_index in numpy.random.permutation(xrange(len(self._corpus[test_document_index]))):
						# get the word_id of at the word_index of the document_index
						word_id = self._corpus[test_document_index][test_word_index]
						
						# retrieve the old_table_id of the current word of current document
						table_id = proposed_t_dv[test_document_index][test_word_index]
						# retrieve the old_topic_id of the table that current word of current document sit on
						topic_id = proposed_k_dt[test_document_index][table_id]
		
						test_n_dt[table_id] += 1
						test_n_dk[topic_id] += 1                
						test_n_kv[topic_id, word_id] += 1
						
					for test_table_index in xrange(len(test_n_dt)):
						assert test_n_dt[test_table_index] == proposed_n_dt[test_document_index][test_table_index], (test_n_dt, proposed_n_dt[test_document_index])
					
					assert numpy.all(test_n_dk == proposed_n_dk[test_document_index, :]), (test_n_dk, proposed_n_dk[test_document_index, :], test_document_index)
				
				assert numpy.all(test_n_kv == proposed_n_kv)
				'''

				if numpy.random.random() < 0.5:
					proposed_k_dt[document_index][table_index] = proposed_K - 1

					proposed_m_k[component_index] -= 1
					proposed_m_k[proposed_K - 1] += 1

					proposed_n_dk[document_index, component_index] -= proposed_n_dt[document_index][table_index]
					proposed_n_dk[document_index, proposed_K - 1] += proposed_n_dt[document_index][table_index]

					selected_word_index = numpy.nonzero(proposed_t_dv[document_index] == table_index)[0]
					selected_word_freq_dist = nltk.probability.FreqDist(
						[self._corpus[document_index][term] for term in list(selected_word_index)])
					for word_index in selected_word_freq_dist:
						proposed_n_kv[component_index, word_index] -= selected_word_freq_dist[word_index]
						proposed_n_kv[proposed_K - 1, word_index] += selected_word_freq_dist[word_index]

				number_of_unvisited_target_tables -= 1

				if number_of_unvisited_target_tables == 0:
					break

			if number_of_unvisited_target_tables == 0:
				break

		if proposed_m_k[component_index] == 0 or proposed_m_k[proposed_K - 1] == 0:
			return None
		else:
			model_parameter = (
				proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
			# self.model_assertion(model_parameter), proposed_m_k
			return model_parameter

	def sequential_allocation_split(self, component_index, model_parameter):
		# sample the data points set
		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		document_table_indices = []
		for document_index in xrange(self._D):
			for table_index in xrange(len(proposed_k_dt[document_index])):
				if proposed_k_dt[document_index][table_index] == component_index:
					document_table_indices.append((document_index, table_index))

				if len(document_table_indices) == proposed_m_k[component_index]:
					break
			if len(document_table_indices) == proposed_m_k[component_index]:
				break

		assert len(document_table_indices) == proposed_m_k[component_index]

		if len(document_table_indices) < 2:
			return None

		# randomly choose two points and initialize the cluster
		random.shuffle(document_table_indices)

		# clear current cluster
		proposed_n_kv[component_index, :] = 0
		proposed_m_k[component_index] = 0
		proposed_n_dk[:, component_index] = 0

		# create a new cluster
		proposed_n_kv = numpy.vstack((proposed_n_kv, numpy.zeros((1, self._vocabulary_size), dtype=numpy.int)))
		proposed_m_k = numpy.hstack((proposed_m_k, numpy.zeros(1, dtype=numpy.int)))
		proposed_n_dk = numpy.hstack((proposed_n_dk, numpy.zeros((self._D, 1), dtype=numpy.int)))
		proposed_K += 1

		# initialize the existing cluster
		(document_index_1, table_index_1) = document_table_indices.pop()
		proposed_k_dt[document_index_1][table_index_1] = component_index
		proposed_m_k[component_index] = 1
		proposed_n_dk[document_index_1, component_index] = proposed_n_dt[document_index_1][table_index_1]

		selected_word_index = numpy.nonzero(proposed_t_dv[document_index_1] == table_index_1)[0]
		# find the frequency distribution of the words sitting on the current table
		selected_word_freq_dist = nltk.probability.FreqDist(
			[self._corpus[document_index_1][term] for term in list(selected_word_index)])
		for word_id in selected_word_freq_dist:
			proposed_n_kv[component_index, word_id] = selected_word_freq_dist[word_id]

		# initialize the new cluster
		(document_index_2, table_index_2) = document_table_indices.pop()
		proposed_k_dt[document_index_2][table_index_2] = proposed_K - 1
		proposed_m_k[proposed_K - 1] = 1
		proposed_n_dk[document_index_2, proposed_K - 1] = proposed_n_dt[document_index_2][table_index_2]

		selected_word_index = numpy.nonzero(proposed_t_dv[document_index_2] == table_index_2)[0]
		# find the frequency distribution of the words sitting on the current table
		selected_word_freq_dist = nltk.probability.FreqDist(
			[self._corpus[document_index_2][term] for term in list(selected_word_index)])
		for word_id in selected_word_freq_dist:
			proposed_n_kv[proposed_K - 1, word_id] = selected_word_freq_dist[word_id]

		# sequentially allocation all the rest points to different clusters
		n_k = numpy.sum(proposed_n_kv, axis=1)
		for (document_index, table_index) in document_table_indices:
			selected_word_index = numpy.nonzero(proposed_t_dv[document_index] == table_index)[0]
			# find the frequency distribution of the words sitting on the current table
			selected_word_freq_dist = nltk.probability.FreqDist(
				[self._corpus[document_index][term] for term in list(selected_word_index)])

			# compute the probability of being in current cluster
			# current_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
			current_topic_probability = scipy.special.gammaln(
				self._vocabulary_size * self._alpha_eta + n_k[component_index])
			current_topic_probability -= scipy.special.gammaln(
				self._vocabulary_size * self._alpha_eta + n_k[component_index] + proposed_n_dt[document_index][
					table_index])
			for word_id in selected_word_freq_dist:
				current_topic_probability += scipy.special.gammaln(
					selected_word_freq_dist[word_id] + proposed_n_kv[component_index, word_id] + self._alpha_eta)
				current_topic_probability -= scipy.special.gammaln(
					proposed_n_kv[component_index, word_id] + self._alpha_eta)
			# current_topic_probability -= scipy.special.gammaln(self._alpha_eta)
			current_topic_probability += numpy.log(proposed_m_k[component_index])

			# compute the probability of being in other cluster
			# other_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
			other_topic_probability = scipy.special.gammaln(
				self._vocabulary_size * self._alpha_eta + n_k[proposed_K - 1])
			other_topic_probability -= scipy.special.gammaln(
				self._vocabulary_size * self._alpha_eta + n_k[proposed_K - 1] + proposed_n_dt[document_index][
					table_index])
			for word_id in selected_word_freq_dist:
				other_topic_probability += scipy.special.gammaln(
					proposed_n_kv[proposed_K - 1, word_id] + self._alpha_eta + selected_word_freq_dist[word_id])
				other_topic_probability -= scipy.special.gammaln(
					proposed_n_kv[proposed_K - 1, word_id] + self._alpha_eta)
			# other_topic_probability -= scipy.special.gammaln(self._alpha_eta)
			other_topic_probability += numpy.log(proposed_m_k[proposed_K - 1])

			# sample a new cluster label for current table
			ratio_other_over_current = numpy.exp(other_topic_probability - current_topic_probability)
			cluster_probability_current = 1. / (1. + ratio_other_over_current)
			if numpy.random.random() <= cluster_probability_current:
				new_label = component_index
			else:
				new_label = proposed_K - 1

			# update the cluster parameters
			proposed_k_dt[document_index][table_index] = new_label
			proposed_m_k[new_label] += 1
			proposed_n_dk[document_index, new_label] += proposed_n_dt[document_index][table_index]
			for word_id in selected_word_freq_dist:
				proposed_n_kv[new_label, word_id] += selected_word_freq_dist[word_id]

			n_k[new_label] += selected_word_freq_dist.N()

		assert proposed_m_k[component_index] > 0 and proposed_m_k[proposed_K - 1] > 0

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		return model_parameter

	def restrict_gibbs_sampling(self, cluster_index_1, cluster_index_2, model_parameter,
	                            restricted_gibbs_sampling_iteration=1):
		# self.model_assertion(model_parameter)
		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		document_table_indices = []
		for document_index in xrange(self._D):
			for table_index in xrange(len(proposed_k_dt[document_index])):
				if proposed_k_dt[document_index][table_index] == cluster_index_1 or proposed_k_dt[document_index][
					table_index] == cluster_index_2:
					document_table_indices.append((document_index, table_index))

				if len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2]:
					break

			if len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2]:
				break

		assert len(document_table_indices) == proposed_m_k[cluster_index_1] + proposed_m_k[cluster_index_2], (
			len(document_table_indices), proposed_m_k, cluster_index_1, cluster_index_2)

		for restricted_gibbs_sampling_iteration_index in xrange(restricted_gibbs_sampling_iteration):
			transition_log_probability = 0
			for (document_index, table_index) in numpy.random.permutation(document_table_indices):
				current_topic_id = proposed_k_dt[document_index][table_index]

				if current_topic_id == cluster_index_1:
					other_topic_id = cluster_index_2
				elif current_topic_id == cluster_index_2:
					other_topic_id = cluster_index_1
				else:
					sys.stderr.write("error: table does not belong to proposed split clusters...\n")

				# find the index of the words sitting on the current table
				selected_word_index = numpy.nonzero(proposed_t_dv[document_index] == table_index)[0]
				# find the frequency distribution of the words sitting on the current table
				selected_word_freq_dist = nltk.probability.FreqDist(
					[self._corpus[document_index][term] for term in list(selected_word_index)])

				n_k = numpy.sum(proposed_n_kv, axis=1)

				if proposed_m_k[current_topic_id] <= 1:
					# if current table is the only table assigned to current topic,
					current_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
					current_topic_probability -= scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + proposed_n_dt[document_index][table_index])
					for word_id in selected_word_freq_dist:
						current_topic_probability += scipy.special.gammaln(
							selected_word_freq_dist[word_id] + self._alpha_eta)
						current_topic_probability -= scipy.special.gammaln(self._alpha_eta)
					current_topic_probability += numpy.log(self._alpha_alpha)
				else:
					# if there are other tables assigned to current topic
					# current_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
					current_topic_probability = scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + n_k[current_topic_id] - proposed_n_dt[document_index][
							table_index])
					current_topic_probability -= scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + n_k[current_topic_id])
					for word_id in selected_word_freq_dist:
						current_topic_probability += scipy.special.gammaln(
							proposed_n_kv[current_topic_id, word_id] + self._alpha_eta)
						current_topic_probability -= scipy.special.gammaln(
							proposed_n_kv[current_topic_id, word_id] + self._alpha_eta - selected_word_freq_dist[
								word_id])
					# current_topic_probability -= scipy.special.gammaln(self._alpha_eta)
					# compute the prior if we move this table from this topic
					current_topic_probability += numpy.log(proposed_m_k[current_topic_id] - 1)

				if proposed_m_k[other_topic_id] <= 0:
					# if current table is the only table assigned to current topic,
					other_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
					other_topic_probability -= scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + proposed_n_dt[document_index][table_index])
					for word_id in selected_word_freq_dist:
						other_topic_probability += scipy.special.gammaln(
							selected_word_freq_dist[word_id] + self._alpha_eta)
						other_topic_probability -= scipy.special.gammaln(self._alpha_eta)
					other_topic_probability += numpy.log(self._alpha_alpha)
				else:
					other_topic_probability = scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + n_k[other_topic_id])
					# other_topic_probability = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
					other_topic_probability -= scipy.special.gammaln(
						self._vocabulary_size * self._alpha_eta + n_k[other_topic_id] + proposed_n_dt[document_index][
							table_index])
					for word_id in selected_word_freq_dist:
						other_topic_probability += scipy.special.gammaln(
							proposed_n_kv[other_topic_id, word_id] + self._alpha_eta + selected_word_freq_dist[
								word_id])
						other_topic_probability -= scipy.special.gammaln(
							proposed_n_kv[other_topic_id, word_id] + self._alpha_eta)
					# other_topic_probability -= scipy.special.gammaln(self._alpha_eta)
					other_topic_probability += numpy.log(proposed_m_k[other_topic_id])

				# sample a new cluster label for current point
				ratio_current_over_other = numpy.exp(current_topic_probability - other_topic_probability)
				probability_other_topic = 1. / (1. + ratio_current_over_other)

				# if this table does not change topic assignment
				if numpy.random.random() > probability_other_topic:
					transition_log_probability += numpy.log(1 - probability_other_topic)
					continue

				transition_log_probability += numpy.log(probability_other_topic)

				# assign this table to new topic
				proposed_k_dt[document_index][table_index] = other_topic_id

				# adjust the statistics of all model parameter
				proposed_m_k[current_topic_id] -= 1
				proposed_m_k[other_topic_id] += 1
				proposed_n_dk[document_index, current_topic_id] -= proposed_n_dt[document_index][table_index]
				proposed_n_dk[document_index, other_topic_id] += proposed_n_dt[document_index][table_index]
				for word_id in selected_word_freq_dist:
					proposed_n_kv[current_topic_id, word_id] -= selected_word_freq_dist[word_id]
					assert (proposed_n_kv[current_topic_id, word_id] >= 0)
					proposed_n_kv[other_topic_id, word_id] += selected_word_freq_dist[word_id]

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		# self.model_assertion(model_parameter)
		return model_parameter, transition_log_probability

	def merge_metropolis_hastings(self, component_index_1, component_index_2):
		old_log_posterior = self.log_posterior()

		# this is to switch the label, make sure we always
		if component_index_1 > component_index_2:
			temp_random_label = component_index_1
			component_index_1 = component_index_2
			component_index_2 = temp_random_label

		proposed_K = self._K

		proposed_n_kv = numpy.copy(self._n_kv)
		proposed_m_k = numpy.copy(self._m_k)
		proposed_n_dk = numpy.copy(self._n_dk)

		proposed_n_dt = copy.deepcopy(self._n_dt)

		proposed_t_dv = copy.deepcopy(self._t_dv)
		proposed_k_dt = copy.deepcopy(self._k_dt)

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		# self.model_assertion(model_parameter)

		if self._merge_proposal == 0:
			# perform random merge for merge proposal
			model_parameter = self.random_merge(component_index_1, component_index_2, model_parameter)
			# self.model_assertion(model_parameter)

			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

			log_proposal_probability = -(proposed_m_k[component_index_1] - 2) * numpy.log(2)
		elif self._merge_proposal == 1:
			# perform restricted gibbs sampling for merge proposal
			model_parameter, transition_log_probability = self.restrict_gibbs_sampling(component_index_1,
			                                                                           component_index_2,
			                                                                           model_parameter,
			                                                                           self._restrict_gibbs_sampling_iteration + 1)
			# self.model_assertion(model_parameter)

			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

			if proposed_m_k[component_index_1] == 0 or proposed_m_k[component_index_2] == 0:
				if proposed_m_k[component_index_1] == 0:
					collapsed_cluster = component_index_1
				elif proposed_m_k[component_index_2] == 0:
					collapsed_cluster = component_index_2

				# since one cluster is empty now, switch it with the last one
				proposed_n_kv[collapsed_cluster, :] = proposed_n_kv[proposed_K - 1, :]
				proposed_m_k[collapsed_cluster] = proposed_m_k[proposed_K - 1]
				proposed_n_dk[:, collapsed_cluster] = proposed_n_dk[:, proposed_K - 1]

				for document_index in xrange(self._D):
					proposed_k_dt[document_index][
						numpy.nonzero(proposed_k_dt[document_index] == (proposed_K - 1))] = collapsed_cluster

				# remove the very last empty cluster, to remain compact cluster
				proposed_n_kv = numpy.delete(proposed_n_kv, [proposed_K - 1], axis=0)
				proposed_m_k = numpy.delete(proposed_m_k, [proposed_K - 1], axis=0)
				proposed_n_dk = numpy.delete(proposed_n_dk, [proposed_K - 1], axis=1)
				proposed_K -= 1

			log_proposal_probability = transition_log_probability
		elif self._merge_proposal == 2:
			# perform gibbs sampling for merge proposal
			cluster_log_probability = numpy.log(proposed_m_k)
			cluster_log_probability = numpy.sum(cluster_log_probability) - cluster_log_probability
			cluster_log_probability -= scipy.misc.logsumexp(cluster_log_probability)

			# choose a cluster that is inverse proportional to its size
			cluster_probability = numpy.exp(cluster_log_probability)
			temp_cluster_probability = numpy.random.multinomial(1, cluster_probability)[numpy.newaxis, :]
			cluster_label = numpy.nonzero(temp_cluster_probability == 1)[1][0]

			model_parameter = self.gibbs_sampling_merge(cluster_label, model_parameter)
			if model_parameter == None:
				return
			# self.model_assertion(model_parameter)

			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

			self._K = proposed_K

			self._n_kv = proposed_n_kv
			self._m_k = proposed_m_k
			self._n_dk = proposed_n_dk

			self._n_dt = proposed_n_dt

			self._t_dv = proposed_t_dv
			self._k_dt = proposed_k_dt

			return
		else:
			sys.stderr.write("error: unrecognized merge proposal strategy %d...\n" % (self._merge_proposal))

		new_log_posterior = self.log_posterior(model_parameter)

		acceptance_log_probability = log_proposal_probability + new_log_posterior - old_log_posterior
		acceptance_probability = numpy.exp(acceptance_log_probability)

		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		if numpy.random.random() < acceptance_probability:
			print "merge operation granted from %s to %s with acceptance probability %s" % (
				self._m_k, proposed_m_k, acceptance_probability)

			self._K = proposed_K

			self._n_kv = proposed_n_kv
			self._m_k = proposed_m_k
			self._n_dk = proposed_n_dk

			self._n_dt = proposed_n_dt

			self._t_dv = proposed_t_dv
			self._k_dt = proposed_k_dt

		# self.model_assertion()

	def random_merge(self, component_index_1, component_index_2, model_parameter):
		assert component_index_2 > component_index_1
		# self.model_assertion(model_parameter)

		# sample the data points set
		(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
		 proposed_k_dt) = model_parameter

		if component_index_2 == proposed_K - 1:
			number_of_unvisited_target_tables = proposed_m_k[component_index_2]
		else:
			number_of_unvisited_target_tables = proposed_m_k[component_index_2] + proposed_m_k[proposed_K - 1]

		# for document_index in numpy.random.permutation(xrange(self._D)):
		for document_index in xrange(self._D):
			# for table_index in numpy.random.permutation(xrange(len(proposed_k_dt[document_index]))):
			for table_index in xrange(len(proposed_k_dt[document_index])):
				if proposed_k_dt[document_index][table_index] == component_index_2:
					# merge component_index_2 with component_index_1
					proposed_k_dt[document_index][table_index] = component_index_1
					number_of_unvisited_target_tables -= 1

				if proposed_k_dt[document_index][table_index] == proposed_K - 1:
					# shift the very last component to component_index_2
					proposed_k_dt[document_index][table_index] = component_index_2
					number_of_unvisited_target_tables -= 1

				if number_of_unvisited_target_tables == 0:
					break

			if number_of_unvisited_target_tables == 0:
				break

		# merge component_index_2 with component_index_1
		proposed_n_kv[component_index_1, :] += proposed_n_kv[component_index_2, :]
		proposed_m_k[component_index_1] += proposed_m_k[component_index_2]
		proposed_n_dk[:, component_index_1] += proposed_n_dk[:, component_index_2]

		# shift the very last component to component_index_2
		proposed_n_kv[component_index_2, :] = proposed_n_kv[proposed_K - 1, :]
		proposed_m_k[component_index_2] = proposed_m_k[proposed_K - 1]
		proposed_n_dk[:, component_index_2] = proposed_n_dk[:, proposed_K - 1]

		# remove the very last component
		proposed_n_kv = numpy.delete(proposed_n_kv, [proposed_K - 1], axis=0)
		proposed_m_k = numpy.delete(proposed_m_k, [proposed_K - 1], axis=0)
		proposed_n_dk = numpy.delete(proposed_n_dk, [proposed_K - 1], axis=1)
		proposed_K -= 1

		model_parameter = (
			proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv, proposed_k_dt)
		# self.model_assertion(model_parameter)

		return model_parameter

	def gibbs_sampling_merge(self, component_label, model_parameter):
		# self.model_assertion(model_parameter)

		new_label = self.propose_component_to_merge(component_label, model_parameter)

		# compute the prior of being in any of the clusters
		if new_label != component_label:
			# always merge the later cluster to the earlier cluster
			# this is to avoid errors if new_label is the last cluster
			if new_label > component_label:
				temp_label = component_label
				component_label = new_label
				new_label = temp_label

			model_parameter = self.random_merge(new_label, component_label, model_parameter)
			# self.model_assertion(model_parameter)
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter
			print "merge operation granted from %s to %s" % (self._m_k, proposed_m_k)

			return model_parameter
		else:
			return None

	def propose_component_to_merge(self, component_label, model_parameter=None):
		if model_parameter == None:
			proposed_K = self._K
			proposed_n_kv = self._n_kv
			proposed_m_k = self._m_k
			proposed_n_dk = self._n_dk
			proposed_n_dt = self._n_dt
			proposed_t_dv = self._t_dv
			proposed_k_dt = self._k_dt
		else:
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

		# compute the prior of being in any of the clusters
		temp_m_k = numpy.copy(proposed_m_k)
		temp_m_k[component_label] = self._alpha_alpha
		# total_table_count = numpy.sum(proposed_m_k)

		component_log_prior = scipy.special.gammaln(temp_m_k + proposed_m_k[component_label])
		component_log_prior -= scipy.special.gammaln(temp_m_k)

		# adjust for current cluster label
		component_log_prior[component_label] = numpy.log(self._alpha_alpha) + scipy.special.gammaln(
			proposed_m_k[component_label])

		# component_log_prior += scipy.special.gammaln(total_table_count - proposed_m_k[component_label] + self._alpha_alpha)
		# component_log_prior -= scipy.special.gammaln(total_table_count + self._alpha_alpha)

		# compute the likelihood of being in any of the clusters
		n_k = numpy.sum(proposed_n_kv, axis=1)

		component_log_likelihood = numpy.zeros(proposed_K)
		for topic_index in xrange(proposed_K):
			if proposed_m_k[topic_index] == 0:
				component_log_likelihood[topic_index] = negative_infinity
				continue

			# compute the probability of being in current cluster
			if topic_index == component_label:
				component_log_likelihood[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
				component_log_likelihood[topic_index] -= scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[component_label])
				component_log_likelihood[topic_index] += numpy.sum(
					scipy.special.gammaln(proposed_n_kv[component_label, :] + self._alpha_eta))
				component_log_likelihood[topic_index] -= self._vocabulary_size * scipy.special.gammaln(self._alpha_eta)
			else:
				component_log_likelihood[topic_index] = scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[topic_index])
				# component_log_likelihood[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
				component_log_likelihood[topic_index] -= scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[component_label])
				component_log_likelihood[topic_index] += numpy.sum(scipy.special.gammaln(
					proposed_n_kv[topic_index, :] + proposed_n_kv[component_label, :] + self._alpha_eta))
				component_log_likelihood[topic_index] -= numpy.sum(
					scipy.special.gammaln(proposed_n_kv[topic_index, :] + self._alpha_eta))
			# component_log_likelihood[topic_index] -= self._vocabulary_size * scipy.special.gammaln(self._alpha_eta)

		# normalize the posterior distribution
		component_log_posterior = component_log_prior + component_log_likelihood
		component_log_posterior -= scipy.misc.logsumexp(component_log_posterior)
		component_posterior = numpy.exp(component_log_posterior)

		cdf = numpy.cumsum(component_posterior)
		new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
		assert new_label >= 0 and new_label < proposed_K

		return new_label

	def old_propose_component_to_merge(self, component_label, model_parameter=None):
		if model_parameter == None:
			proposed_K = self._K
			proposed_n_kv = self._n_kv
			proposed_m_k = self._m_k
			proposed_n_dk = self._n_dk
			proposed_n_dt = self._n_dt
			proposed_t_dv = self._t_dv
			proposed_k_dt = self._k_dt
		else:
			(proposed_K, proposed_n_kv, proposed_m_k, proposed_n_dk, proposed_n_dt, proposed_t_dv,
			 proposed_k_dt) = model_parameter

		# compute the prior of being in any of the clusters
		temp_m_k = numpy.copy(proposed_m_k)
		temp_m_k[component_label] = self._alpha_alpha
		total_table_count = numpy.sum(proposed_m_k)

		component_log_prior = scipy.special.gammaln(temp_m_k + proposed_m_k[component_label])
		component_log_prior -= scipy.special.gammaln(temp_m_k)

		component_log_prior += scipy.special.gammaln(
			total_table_count - proposed_m_k[component_label] + self._alpha_alpha)
		component_log_prior -= scipy.special.gammaln(total_table_count + self._alpha_alpha)

		# compute the likelihood of being in any of the clusters
		n_k = numpy.sum(proposed_n_kv, axis=1)

		component_log_likelihood = numpy.zeros(proposed_K)
		for topic_index in xrange(proposed_K):
			if proposed_m_k[topic_index] == 0:
				component_log_likelihood[topic_index] = negative_infinity
				continue

			# compute the probability of being in current cluster
			if topic_index == component_label:
				component_log_likelihood[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
				component_log_likelihood[topic_index] -= scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[component_label])
				component_log_likelihood[topic_index] += numpy.sum(
					scipy.special.gammaln(proposed_n_kv[component_label, :] + self._alpha_eta))
				component_log_likelihood[topic_index] -= self._vocabulary_size * scipy.special.gammaln(self._alpha_eta)
			else:
				component_log_likelihood[topic_index] = scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[topic_index])
				# component_log_likelihood[topic_index] = scipy.special.gammaln(self._vocabulary_size * self._alpha_eta)
				component_log_likelihood[topic_index] -= scipy.special.gammaln(
					self._vocabulary_size * self._alpha_eta + n_k[topic_index] + n_k[component_label])
				component_log_likelihood[topic_index] += numpy.sum(scipy.special.gammaln(
					proposed_n_kv[topic_index, :] + proposed_n_kv[component_label, :] + self._alpha_eta))
				component_log_likelihood[topic_index] -= numpy.sum(
					scipy.special.gammaln(proposed_n_kv[topic_index, :] + self._alpha_eta))
			# component_log_likelihood[topic_index] -= self._vocabulary_size * scipy.special.gammaln(self._alpha_eta)

		# normalize the posterior distribution
		component_log_posterior = component_log_prior + component_log_likelihood
		component_log_posterior -= scipy.misc.logsumexp(component_log_posterior)
		component_posterior = numpy.exp(component_log_posterior)

		cdf = numpy.cumsum(component_posterior)
		new_label = numpy.uint(numpy.nonzero(cdf >= numpy.random.random())[0][0])
		assert new_label >= 0 and new_label < proposed_K

		return new_label

	def resample_components(self):
		if self._K == 1:
			return

		# self.model_assertion()

		assert numpy.all(self._m_k >= 0)

		# sample cluster assignment for all the points in the current cluster
		# for old_component_label in numpy.random.permutation(self._K):
		for old_component_label in numpy.argsort(self._m_k):
			# if this cluster is empty, no need to resample the cluster assignment
			if self._m_k[old_component_label] <= 0:
				continue

			new_component_label = self.propose_component_to_merge(old_component_label)
			# print "propose to merge cluster %d to %d" % (old_component_label, new_component_label)

			if new_component_label == old_component_label:
				continue

			# find the index of the data point in the current cluster
			candidate_table_counter = 0
			for document_index in xrange(self._D):
				for table_index in xrange(len(self._k_dt[document_index])):
					if self._k_dt[document_index][table_index] == old_component_label:
						self._k_dt[document_index][table_index] = new_component_label
						candidate_table_counter += 1

					if candidate_table_counter == self._m_k[old_component_label]:
						break
				if candidate_table_counter == self._m_k[old_component_label]:
					break

			self._m_k[new_component_label] += self._m_k[old_component_label]
			self._m_k[old_component_label] = 0
			self._n_kv[new_component_label, :] += self._n_kv[old_component_label, :]
			self._n_kv[old_component_label, :] = 0
			self._n_dk[:, new_component_label] += self._n_dk[:, old_component_label]
			self._n_dk[:, old_component_label] = 0

			print "merge cluster %d to %d after component resampling to %s..." % (
				old_component_label, new_component_label, self._m_k)

		# self.model_assertion()

		self.compact_params()

		'''
		empty_topics = numpy.nonzero(self._m_k == 0)[0]
		non_empty_clusters = numpy.nonzero(self._m_k > 0)[0]
		
		# shift down all the cluster indices
		for document_index in xrange(self._D):
			for component_label in xrange(len(non_empty_clusters)):
				self._k_dt[document_index][numpy.nonzero(self._k_dt[document_index] == non_empty_clusters[component_label])[0]] = component_label
		
		self._n_kv[new_component_label, :] += self._n_kv[old_component_label, :]
		self._n_kv[old_component_label, :] = 0
		self._n_dk[:, new_component_label] += self._n_dk[:, old_component_label]
		self._n_dk[:, old_component_label] = 0
		
		self._K -= len(empty_topics)
		
		self._m_k = numpy.delete(self._m_k, empty_topics, axis=0)
		assert self._m_k.shape == (self._K,)
		self._n_kv = numpy.delete(self._n_kv, empty_topics, axis=0)
		assert self._n_kv.shape == (self._K, self._vocabulary_size)
		self._n_dk = numpy.delete(self._n_dk, empty_topics, axis=1)
		assert self._n_dk.shape == (self._D, self._K)
		'''

		# self.model_assertion()

		return

	"""
	"""

	def compact_params(self, model_parameter=None):
		# self.model_assertion(model_parameter)

		if model_parameter == None:
			K = self._K
			n_kv = self._n_kv
			m_k = self._m_k
			n_dk = self._n_dk
			n_dt = self._n_dt
			t_dv = self._t_dv
			k_dt = self._k_dt
		else:
			(K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter

		# find unused and used topics
		unused_topics = numpy.nonzero(m_k == 0)[0]
		used_topics = numpy.nonzero(m_k != 0)[0]

		K -= len(unused_topics)
		assert (K >= 1 and K == len(used_topics))

		n_dk = numpy.delete(n_dk, unused_topics, axis=1)
		assert (n_dk.shape == (self._D, K))
		n_kv = numpy.delete(n_kv, unused_topics, axis=0)
		assert (n_kv.shape == (K, self._vocabulary_size))
		m_k = numpy.delete(m_k, unused_topics)
		assert (len(m_k) == K)

		for d in xrange(self._D):
			# find the unused and used tables
			unused_tables = numpy.nonzero(n_dt[d] == 0)[0]
			used_tables = numpy.nonzero(n_dt[d] != 0)[0]

			n_dt[d] = numpy.delete(n_dt[d], unused_tables)
			k_dt[d] = numpy.delete(k_dt[d], unused_tables)

			# shift down all the table indices of all words in current document
			# @attention: shift the used tables in ascending order only.
			for t in xrange(len(n_dt[d])):
				t_dv[d][numpy.nonzero(t_dv[d] == used_tables[t])[0]] = t

			# shrink down all the topics indices of all tables in current document
			# @attention: shrink the used topics in ascending order only.
			for k in xrange(K):
				k_dt[d][numpy.nonzero(k_dt[d] == used_topics[k])[0]] = k

		if model_parameter == None:
			self._K = K
			self._n_kv = n_kv
			self._m_k = m_k
			self._n_dk = n_dk
			self._n_dt = n_dt
			self._t_dv = t_dv
			self._k_dt = k_dt

			# self.model_assertion()
			return
		else:
			model_parameter = (K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt)

			# self.model_assertion(model_parameter)
			return model_parameter

	def log_posterior(self, model_parameter=None, hyper_parameter=None):
		log_posterior = 0.

		if hyper_parameter == None:
			alpha_alpha = self._alpha_alpha
			alpha_gamma = self._alpha_gamma
			alpha_eta = self._alpha_eta
		else:
			alpha_alpha = hyper_parameter[0]
			alpha_gamma = hyper_parameter[1]
			alpha_eta = hyper_parameter[2]

		# compute the document level log likelihood
		log_posterior += self.table_log_likelihood(model_parameter, alpha_gamma)
		# compute the table level log likelihood
		log_posterior += self.topic_log_likelihood(model_parameter, alpha_alpha)
		# compute the word level log likelihood
		log_posterior += self.word_log_likelihood(model_parameter, alpha_eta)

		return log_posterior

	"""
	compute the word level log likelihood p(x | t, k) = \prod_{k=1}^K f(x_{ij} | z_{ij}=k), where f(x_{ij} | z_{ij}=k) = \frac{\Gamma(V \eta)}{\Gamma(n_k + V \eta)} \frac{\prod_{v} \Gamma(n_{k}^{v} + \eta)}{\Gamma^V(\eta)}
	"""

	def word_log_likelihood(self, model_parameter=None, alpha_eta=None):
		if model_parameter == None:
			K = self._K
			n_kv = self._n_kv
			m_k = self._m_k
			n_dk = self._n_dk
			n_dt = self._n_dt
			t_dv = self._t_dv
			k_dt = self._k_dt
		else:
			(K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter

		if alpha_eta == None:
			alpha_eta = self._alpha_eta

		n_k = numpy.sum(n_kv, axis=1)
		assert (len(n_k) == K)

		log_likelihood = 0

		log_likelihood += K * scipy.special.gammaln(self._vocabulary_size * alpha_eta)
		log_likelihood -= numpy.sum(scipy.special.gammaln(self._vocabulary_size * alpha_eta + n_k))

		log_likelihood += numpy.sum(scipy.special.gammaln(alpha_eta + n_kv))
		log_likelihood -= K * self._vocabulary_size * scipy.special.gammaln(alpha_eta)

		return log_likelihood

	"""
	compute the table level prior in log scale \prod_{d=1}^D (p(t_{d})), where p(t_d) = \frac{ \alpha^m_d \prod_{t=1}^{m_d}(n_di-1)! }{ \prod_{v=1}^{n_d}(v+\alpha-1) }
	"""

	def table_log_likelihood(self, model_parameter=None, alpha_gamma=None):
		if model_parameter == None:
			K = self._K
			n_kv = self._n_kv
			m_k = self._m_k
			n_dk = self._n_dk
			n_dt = self._n_dt
			t_dv = self._t_dv
			k_dt = self._k_dt
		else:
			(K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter

		if alpha_gamma == None:
			alpha_gamma = self._alpha_gamma

		log_likelihood = 0.
		for document_index in xrange(self._D):
			log_likelihood += len(k_dt[document_index]) * numpy.log(alpha_gamma)
			log_likelihood += numpy.sum(scipy.special.gammaln(n_dt[document_index]))
			log_likelihood -= scipy.special.gammaln(len(t_dv[document_index]) + alpha_gamma)
			log_likelihood += scipy.special.gammaln(alpha_gamma)

		return log_likelihood

	"""
	compute the topic level prior in log scale p(k) = \frac{ \gamma^K \prod_{k=1}^{K}(m_k-1)! }{ \prod_{s=1}^{m}(s+\gamma-1) }
	"""

	def topic_log_likelihood(self, model_parameter=None, alpha_alpha=None):
		if model_parameter == None:
			K = self._K
			n_kv = self._n_kv
			m_k = self._m_k
			n_dk = self._n_dk
			n_dt = self._n_dt
			t_dv = self._t_dv
			k_dt = self._k_dt
		else:
			(K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter

		if alpha_alpha == None:
			alpha_alpha = self._alpha_alpha

		log_likelihood = 0
		log_likelihood += K * numpy.log(alpha_alpha)
		log_likelihood += numpy.sum(scipy.special.gammaln(m_k))
		log_likelihood -= scipy.special.gammaln(numpy.sum(m_k) + alpha_alpha)
		log_likelihood += scipy.special.gammaln(alpha_alpha)

		return log_likelihood

	"""
	"""

	def export_beta(self, exp_beta_path, top_display=-1):
		n_k_sum_over_v = numpy.sum(self._n_kv, axis=1)[:, numpy.newaxis]
		assert (n_k_sum_over_v.shape == (self._K, 1))
		beta_probability = (self._n_kv + self._alpha_eta) / (n_k_sum_over_v + self._vocabulary_size * self._alpha_eta)

		output = open(exp_beta_path, 'w')
		for topic_index in xrange(self._K):
			output.write("==========\t%d\t==========\n" % (topic_index))

			i = 0
			for word_index in reversed(numpy.argsort(beta_probability[topic_index, :])):
				i += 1
				output.write(
					self._index_to_word[word_index] + "\t" + str(beta_probability[topic_index, word_index]) + "\n")
				if top_display > 0 and i >= top_display:
					break

		output.close()

	def model_assertion(self, model_parameter=None):
		if model_parameter == None:
			K = self._K
			n_kv = self._n_kv
			m_k = self._m_k
			n_dk = self._n_dk
			n_dt = self._n_dt
			t_dv = self._t_dv
			k_dt = self._k_dt
		else:
			(K, n_kv, m_k, n_dk, n_dt, t_dv, k_dt) = model_parameter

		test_n_kv = numpy.zeros((K, self._vocabulary_size), dtype=numpy.int)
		test_m_k = numpy.zeros(K, dtype=numpy.int)

		for document_index in xrange(self._D):
			test_n_dt = [0] * len(k_dt[document_index])
			test_n_dk = numpy.zeros(K, dtype=numpy.int)

			# sample word assignment, see which table it should belong to
			for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
				# get the word_id of at the word_index of the document_index
				word_id = self._corpus[document_index][word_index]

				# retrieve the old_table_id of the current word of current document
				table_id = t_dv[document_index][word_index]
				# retrieve the old_topic_id of the table that current word of current document sit on
				topic_id = k_dt[document_index][table_id]

				test_n_dt[table_id] += 1
				test_n_dk[topic_id] += 1
				test_n_kv[topic_id, word_id] += 1

			for table_index in xrange(len(test_n_dt)):
				assert test_n_dt[table_index] == n_dt[document_index][table_index]
				if n_dt[document_index][table_index] == 0:
					continue
				topic_id = k_dt[document_index][table_index]
				test_m_k[topic_id] += 1

			assert numpy.all(test_n_dk == n_dk[document_index, :]), (test_n_dk, n_dk[document_index, :])

		assert numpy.all(test_m_k == m_k), (test_m_k, m_k)
		assert numpy.all(test_n_kv == n_kv)


"""
run HDP on a synthetic corpus.
"""
if __name__ == '__main__':
	a = numpy.random.random((2, 3))
	print a
	print a / numpy.sum(a, axis=1)[:, numpy.newaxis]
