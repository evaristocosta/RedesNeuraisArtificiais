# Hopfield Network
# Desenvolvido por: Thiago Fellipe Ortiz de Camargo

import numpy as np
import pandas as pd

# calculo de diferencas
# https://stackoverflow.com/questions/14914615/in-python-find-out-number-of-differences-between-two-ordered-lists
def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))


# Hopfield
class hopfield(object):

    def __init__(self, patterns, noise_percentage, pattern_n_row, pattern_n_column, ib, epochs):
        self.patterns = patterns
        self.noise = 1-noise_percentage
        self.nrow = pattern_n_row
        self.ncol = pattern_n_column
        self.fmn = len(patterns)
        self.dim = len(self.patterns[0])
        self.ib = ib
        self.epc = epochs
        self.scape = False

    def noise_attribution(self, patt):
        self.pattern = patt
        self.randM = np.random.rand(self.nrow, self.ncol)
        self.auxA = self.noise > self.randM
        self.auxB = self.noise < self.randM
        self.randM[self.auxA] = 1
        self.randM[self.auxB] = -1
        self.new_patter = self.pattern.reshape(self.nrow, self.ncol)*self.randM
        return self.new_patter.reshape(self.dim, 1)

    def weights(self):
        self.auxW = 0

        for patt in self.patterns:
            self.auxW += patt*patt.reshape(self.dim, 1)

        self.W = ((1/self.dim)*self.auxW) - \
            ((self.fmn/self.dim)*np.zeros((self.dim, self.dim)))

    def run(self):
        self.outputs = pd.DataFrame()
        self.noised_img = pd.DataFrame()
        for patt, _ in zip(self.patterns, range(self.fmn)):
            self.weights()
            self.v_current = self.noise_attribution(patt)
            self.noised_img = pd.concat(
                (self.noised_img, pd.DataFrame(self.v_current).T))
            self.it = 0
            self.scape = False

            while(self.scape == False):
                self.v_past = self.v_current
                self.u = np.dot(self.W, self.v_past)+self.ib
                self.v_current = np.sign(np.tanh(self.u))

                if pd.DataFrame(self.v_current).equals(pd.DataFrame(self.v_past)):
                    self.scape = True

                if(self.it >= self.epc):
                    self.scape = True

                self.it += 1

            self.outputs = pd.concat(
                (self.outputs, pd.DataFrame(self.v_current).T))
