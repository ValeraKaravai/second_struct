import numpy as np
import theano
import theano.tensor as T
from Bio.PDB import *
import os
import pprint

mode = theano.Mode(linker='cvm')


def Initial_mAmino(aminoFile, a, t):
    mAmino = {}
    f = open(aminoFile, "r")
    for i in f:
        i = i.split()
        vect = np.random.randn(t)
        mAmino.update({i[0]: vect})
    return mAmino


def InitialMatrix(input_file, mAmino, t, max_len, count):
    inp = open(input_file, 'r')
    second = {'H': 0, 'B': 1, 'C': 2}
    bool_len = False
    for j, i in enumerate(inp):
        if j == 0:
            if len(i) >= max_len:
                count -= 1
                bool_len = True
            mt_first = np.zeros(shape=(len(i) - 1, t))
            i = i.split('\n')[0]
            for l, k in enumerate(i):
                vect = mAmino[k]
                mt_first[l, ] = vect
        if j == 1:
            mt_second = []
            for k in i:
                if k != 'H' and k != 'B':
                    mt_second.append(second['C'])
                else:
                    mt_second.append(second[k])
    return mt_first[:max_len], mt_second[:max_len], count, bool_len


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

    def mult(self, y):
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

            self.l_r *= self.l_r_d
            print iterat, this_train_loss, self.l_r


def test_softplus():
    print "Softplus"
    file_dssp = os.listdir('parse_file/pdb/')
    pp = pprint.PrettyPrinter()
    aminoFile = "data/amino_acid.txt"
    second_list = []
    amino_list = []
    t = 11
    a = 20
    n_x = t
    n_h = 6
    n_y = 3
    max_len = 90
    n_seq = 30
    count = n_seq
    mAmino = Initial_mAmino(aminoFile, a, t)
    temp = 0
    while count:
        input_file = 'parse_file/pdb/' + file_dssp[temp]
        temp += 1
        print input_file
        mt_first, mt_second, count, bool_len = InitialMatrix(input_file,
                                                             mAmino, t,
                                                             max_len, count)
        if bool_len:
            amino_list.append(mt_first)
            second_list.append(mt_second)
    model = RecurNN(n_x=n_x, n_h=n_h, n_y=n_y,
                    activ='softplus', output='softmax', l_r=0.001,
                    l_r_d=0.999, L1_reg=0, L2_reg=0, init_mom=0.5,
                    final_mom=0.9, mom_switch=5, n_iter=500)
    model.train(amino_list, second_list)
    guess = model.predict(amino_list[0])
    pp.pprint(guess)
    print(second_list[0])

if __name__ == "__main__":
    aminoFile = "data/amino_acid.txt"
    test_softplus()
