import logging
import gensim
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# OUTPUTS saved in compare_wv_resulsts.txt in the same file directory

path = "/home/intern/video_caption/video_captions/data/"

def print_accuracy(model, questions_file):
    print('Evaluating...\n')
    acc = model.accuracy(questions_file)

    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100*float(sem_correct)/sem_total
    print('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))
    
    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc)-1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5,len(acc)-1))
    syn_acc = 100*float(syn_correct)/syn_total
    print('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return (sem_acc, syn_acc)

word_analogies_file = path + 'word_vectors/questions-words.txt'
accuracies = []
print('\nLoading Gensim embeddings')
caption_gs = gensim.models.KeyedVectors.load_word2vec_format(path + "MSR-VTT/" + 'model.vec')
print('Accuracy for Word2Vec:')
accuracies.append(print_accuracy(caption_gs, word_analogies_file))

print('\nLoading FastText embeddings')
caption_ft = gensim.models.KeyedVectors.load_word2vec_format(path + "word_vectors/fastText/" + 'model.vec')
print('Accuracy for FastText (with n-grams):')
accuracies.append(print_accuracy(caption_ft, word_analogies_file))

print('\nLoading FastText embeddings')
wiki_ft = gensim.models.KeyedVectors.load_word2vec_format(path + "word_vectors/" + 'wiki.en.vec')
print('Accuracy for FastText (with n-grams):')
accuracies.append(print_accuracy(wiki_ft, word_analogies_file))

print accuracies