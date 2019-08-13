import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        
        Y_true0 = Y
        Y_pre0 = Y_.tolist()
        
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print 'Results, using embeddings of dimensionality', len(self.embeddings[X[0]])
        print('-------------------')
        print(results)
        
        print('-------------------')
        

        ## true label value VS predict label value
       
        # Y = self.binarizer.transform(Y)
        
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(len(Y_true0)):
            temp = Y_true0[i]
            true_ = temp[0]
            temp = Y_pre0[i]
            pre_ = str(int(temp[1]))
            if true_ == pre_:
                if true_ == '1':
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if true_ == '1':
                    FN = FN + 1
                else:
                    FP = FP + 1
        if (TP+FP) == 0:
            precision = -1
        else:
            precision = TP/(TP+FP)
        
        if (TP+FN) == 0:
            recall = -1
        else:
            recall = TP/(TP+FN)
            
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall/(precision+recall)
        
        
        result_list = [TP,FP,FN,TN,precision,recall,f1]
        print(result_list)
        #print('Ture Positive:',TP)
        #print('False Positive:',FP)
        #print('False Negative:',FN)
        #print('True Negative:',TN)
        #print('Precision:',precision)
        #print('Recall:',recall)
        #print('model accurucy :',result_list)
        #print('f1_score:',2*precision*recall/(precision+recall))

        print('-------------------')
        # return result_list
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed(seed)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        # print(sum(Y_train))

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return self.evaluate(X_test, Y_test)
    



def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {} 
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y
def read_node_label_deleted(filename,deleted_node):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        if vec[0] not in deleted_node:
            X.append(vec[0])
            Y.append(vec[1:])
    fin.close()
    return X, Y

