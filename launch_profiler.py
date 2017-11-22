import cPickle, string, numpy, getopt, sys, random, time, re, pprint
import datetime, os;

import nltk;
import numpy;
import cProfile

def main():
    # parameter set 1
    input_directory="../input/140806-165121-k5-d500-vpk5-wpd30-False"
    
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    '''
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    '''
    
    # Document
    train_docs = [];
    input_doc_stream = open(os.path.join(input_directory, 'doc.dat'), 'r');
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training documents..."
    
    # Vocabulary
    dictionary_file = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(dictionary_file, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (dictionary_file);
    
    # parameter set 2
    alpha_eta = 1.0 / len(vocab);
    alpha_alpha = 1.0;
    alpha_gamma = 1.0;
    
    # parameter set 3
    training_iterations = 5;
    snapshot_interval = training_iterations+1
        
    # parameter set 4
    split_merge_heuristics = 1;
    split_proposal = 0;
    merge_proposal = 0;
    
    import monte_carlo;
    hdp = monte_carlo.MonteCarlo(split_merge_heuristics, split_proposal, merge_proposal);
    hdp._initialize(train_docs, vocab, alpha_alpha, alpha_gamma, alpha_eta)
    
    for iteration in xrange(training_iterations):
        clock = time.time();
        log_likelihood = hdp.learning();
        clock = time.time() - clock;
        #print 'training iteration %d finished in %f seconds: number-of-topics = %d, log-likelihood = %f' % (hdp._iteration_counter, clock, hdp._K, log_likelihood);

    # gamma_path = os.path.join(output_directory, 'gamma.txt');
    # numpy.savetxt(gamma_path, hdp._document_topic_distribution);
    
    # topic_inactive_counts_path = os.path.join(output_directory, "topic_inactive_counts.txt");
    # numpy.savetxt(topic_inactive_counts_path, hdp._topic_inactive_counts);

if __name__ == '__main__':
    main()