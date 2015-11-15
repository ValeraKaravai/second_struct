import numpy as np
import random
import math
import theano
import theano.tensor as T
import os
import datetime
# import theano.tensor as T
global t, a
global mAmino
global mProt  # matrix amino acid

mode = theano.Mode(linker='cvm')


# def InitialMatrix(aminoFile, proteinFile):
#     mAmino = np.zeros(shape=(a, t))
#     f = open(aminoFile, "r")
#     prot = open(proteinFile, "r")
#     str_amino = ''
#     for i in f:
#         mAmino[len(str_amino), ] = np.random.randn(t)
#         str_amino += i.split()[0]
#     mProt = np.zeros(shape=(30, t))

#     count = 0
#     for i in prot:
#         str_prot = i
#         for j in str_prot:
#             mProt[count, ] = mAmino[str_amino.find(j), ]
#             count += 1
#     return mProt


def softplus(x):
    return math.log(1 + math.exp(x))


class RecurNN (object):

    def __init__(self, n_x, n_h, n_y, activ, output,  l_r, l_r_d,
                 L1_reg, L2_reg, init_mom, final_mom, mom_switch, n_iter):
        self.n_x = int(n_x)
        self.n_h = int(n_h)
        self.n_y = int(n_y)
        if activ == "softplus":
            self.activ = T.nnet.softplus
        else:
            print "No"
        self.output = output
        self.l_r = float(l_r)
        self.l_r_d = float(l_r_d)
        self.final_mom = float(final_mom)
        self.init_mom = float(init_mom)
        self.mom_switch = float(mom_switch)
        self.n_iter = float(n_iter)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)

        # self.x = InitialMatrix(aminoFile, proteinFile)
        self.x = T.matrix()

        self.W_xh = theano.shared(value=np.asarray(np.random.uniform(
                                  low=-0.01, high=0.01,
                                  size=(n_x, n_h)), dtype=theano.config.floatX),
                                  name='W_xh')
        self.W_hh = theano.shared(value=np.asarray(np.random.uniform
                                 (low=-0.01, high=0.01, size=(n_h, n_h)),
                                  dtype=theano.config.floatX),
                                  name='W_hh')
        self.W_hy = theano.shared(value=np.asarray(np.random.uniform(
                                  low=-0.01, high=0.01,
                                  size=(n_h, n_y)), dtype=theano.config.floatX),
                                  name='W_hy')
        self.h0 = theano.shared(value=np.zeros(
            (n_h, ), dtype=theano.config.floatX), name='h0')
        self.b_y = theano.shared(value=np.zeros(
            (n_y, ), dtype=theano.config.floatX), name='b_y')
        self.b_h = theano.shared(value=np.zeros(
            (n_h, ), dtype=theano.config.floatX), name='b_h')

        self.params = [self.W_xh, self.W_hh,
                       self.W_hy, self.h0, self.b_h, self.b_y]
        self.updates = {}
        for i in self.params:
            self.updates[i] = theano.shared(value=np.zeros(i.get_value(
                borrow=True).shape, dtype=theano.config.floatX), name='updates')

        def recur_def(x_t, h_tm1):
            h_t = self.activ(T.dot(x_t, self.W_xh) +
                             T.dot(h_tm1, self.W_hh) + self.b_h)
            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, y_t
        [self.h, self.y_pred], _ = theano.scan(
            recur_def, sequences=self.x, outputs_info=[self.h0, None])

        self.L1 = abs(self.W_xh.sum()) + \
            abs(self.W_hh.sum()) + abs(self.W_hy.sum())

        # square of L2 norm
        self.L2_sqr = (self.W_xh ** 2).sum() + \
            (self.W_hh ** 2).sum() + (self.W_hy ** 2).sum()
        if self.output == 'softmax':
            self.y = T.vector(name='y', dtype='int32')
            self.pyx = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.pyx, axis=-1)
            self.loss = lambda y: self.mult(y)
            self.predict_pr = theano.function(
                inputs=[self.x, ], outputs=self.pyx, mode=mode)
            self.predict = theano.function(
                inputs=[self.x, ], outputs=self.y_out, mode=mode)

        self.err = []

    def mult (self, y):
        return -T.mean(T.log(self.pyx)[T.arange(y.shape[0]), y])

    def train(self, x_train, y_train, x_test=None, y_test=None):
        train_x_set = theano.shared(np.asarray(
            x_train, dtype=theano.config.floatX))
        train_y_set = theano.shared(np.asarray(
            y_train, dtype=theano.config.floatX))
        if self.output == 'softmax':
            train_y_set = T.cast(train_y_set, 'int32')

        index = T.lscalar('index')
        lr = T.scalar('lr', dtype=theano.config.floatX)
        moment = T.scalar('moment', dtype=theano.config.floatX)

        cost = self.loss(self.y) + self.L1_reg * \
            self.L1 + self.L2_reg * self.L2_sqr

        comp_train_err = theano.function(inputs=[index, ], outputs=self.loss(
            self.y), givens={self.x: train_x_set[index],
                             self.y: train_y_set[index]}, mode=mode)

        gparams = []
        for i in self.params:
            gparams.append(T.grad(cost, i))

        update = {}

        for i, j in zip(self.params, gparams):
            weight_update = self.updates[i]
            upd = moment * weight_update - lr * j
            update[weight_update] = upd
            update[i] = i + upd

        train_model = theano.function(inputs=[index, lr, moment],
                                      outputs=cost, updates=update, givens={
                                      self.x: train_x_set[index],
                                      self.y: train_y_set[index]},
                                      mode=mode)

        print 'Training model...'
        iterat = 0
        n_train = train_x_set.get_value(borrow=True).shape[0]

        while (iterat < self.n_iter):
            iterat += 1
            for i in xrange(n_train):
                if iterat > self.mom_switch:
                    effect_mom = self.final_mom
                else:
                    effect_mom = self.init_mom
                ex_cost = train_model(i, self.l_r, effect_mom)
            train_losses = [comp_train_err(k) for k in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            self.err.append(this_train_loss)

            print iterat, this_train_loss, self.l_r

            self.l_r *= self.l_r_d


def test_softplus():
    print "Test softplus"

    n_x = 2
    n_h = 6
    n_y = 3
    time_steps = 10
    n_seq = 100

    np.random.seed(0)

    seq = np.random.randn(n_seq, time_steps, n_x)

    tar = np.zeros((n_seq, time_steps), dtype=np.int)

    th = 0.5

    tar[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + th] = 1
    tar[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] + th] = 2

    model = RecurNN(n_x=n_x, n_h=n_h, n_y=n_y,
                    activ='softplus', output='softmax', l_r=0.001,
                    l_r_d=0.999, L1_reg=0, L2_reg=0, init_mom=0.5,
                    final_mom=0.9, mom_switch=5, n_iter=100)
    model.train(seq, tar)

    guess = model.predict_pr(seq[1])


# aminoFile = "amino_acid.txt"
# proteinFile = "protein_2.txt"
# t = 1024
# a = 20
# mProt = InitialMatrix(aminoFile, proteinFile)

if __name__ == "__main__":
    test_softplus()
