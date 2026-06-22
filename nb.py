# coding=utf-8

import sys
from math import log

from common import read_sparse_data

'''
Event model:
0 : Multi-variate Bernoulli event model
1 : Multinomial event model
'''

Bernoulli = 0
Multinomial = 1

class NaiveBayes:
    def __init__(self):
        self.priors = dict()
        self.prob = dict()
    
    def train(self, X, Y, event_model = Bernoulli, smooth = 0.5):

        if event_model != Bernoulli and event_model != Multinomial:
            print('event_model parameter error', file=sys.stderr)
            return

        self.priors.clear()
        self.prob.clear()

        for y in Y:
            if not y in self.priors:
                self.priors[y] = 0
            self.priors[y] += 1

        D = set()
        for x, y in zip(X, Y):
            if not y in self.prob:
                self.prob[y] = dict()
            for w, c in x:
                D.add(w)
                if not w in self.prob[y]:
                    self.prob[y][w] = 0
                
                if event_model == Bernoulli:
                    self.prob[y][w] += 1
                if event_model == Multinomial:
                    self.prob[y][w] += c

        # laplace smooth
        for y in self.prob:
            for w in D:
                if not w in self.prob[y]:
                    self.prob[y][w] = 0
                self.prob[y][w] += smooth

        if event_model == Bernoulli:
            for y in self.prob:
                for w in self.prob[y]:
                    self.prob[y][w] /= 1.0 * self.priors[y]
        
        if event_model == Multinomial:
            for y in self.prob:
                n = sum(self.prob[y].values())
                for w in self.prob[y]:
                    self.prob[y][w] /= 1.0 * n
            
        n = len(Y)
        for y in self.priors:
            self.priors[y] /= 1.0 * n

        print('-' * 128, file=sys.stderr)
        print('Top 5 strong features for each class:', file=sys.stderr)
        print('-' * 128, file=sys.stderr)
        
        for y in self.prob:
            ftrs = sorted(self.prob[y].items(), key = lambda x : x[1], reverse = True)
            print(y + '\t' + '\t'.join([k + ':' + str(round(v, 5)) for k,v in ftrs[0 : 5]]), file=sys.stderr)

        print('-' * 128, file=sys.stderr)

    def test(self, X, Y):
        
        correct = 0
        
        for i, x in enumerate(X):
            d = dict()
            for y in self.priors:
                d[y] = log(self.priors[y])
                for w, c in x:
                    if not w in self.prob[y]:
                        continue
                    d[y] += log(self.prob[y][w])
            prob_list = sorted(d.items(), key = lambda x : x[1], reverse = True)
            
            y_pred = prob_list[0][0]     
    
            if y_pred == Y[i]:
                correct += 1
        
        return 1.0 * correct / len(Y)


if __name__ == '__main__':

    train_path = 'data/20_newsgroups.train'
    test_path = 'data/20_newsgroups.test'

    X_train, Y_train = read_sparse_data(open(train_path))
    X_test, Y_test = read_sparse_data(open(test_path))

    clf = NaiveBayes()
    clf.train(X_train, Y_train, Bernoulli)
    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print('Training accuracy for Multi-variate Bernoulli event model : %f%%' % (100 * acc_train), file=sys.stderr)
    print('Test accuracy for Multi-variate Bernoulli event model : %f%%' % (100 * acc_test), file=sys.stderr)

    clf.train(X_train, Y_train, Multinomial)
    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print('Training accuracy for Multinomial event model : %f%%' % (100 * acc_train), file=sys.stderr)
    print('Test accuracy for Multinomial event model : %f%%' % (100 * acc_test), file=sys.stderr)

