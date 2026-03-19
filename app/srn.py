import numpy as np

class SRN:

    def __init__(self, vocab_size, hidden_size=20, lr=0.05, mu=0, clearval=0.5):

        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size

        self.lr = lr
        self.mu = mu
        self.clearval = clearval

        self.Wxh = np.random.uniform(-0.1,0.1,(hidden_size,vocab_size))
        self.Whh = np.random.uniform(-0.1,0.1,(hidden_size,hidden_size))
        self.Why = np.random.uniform(-0.1,0.1,(vocab_size,hidden_size))

        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)

        self.reset_context()

    def reset_context(self):
        self.context = np.ones(self.hidden_size) * self.clearval

    def softmax(self,x):
        e = np.exp(x-np.max(x))
        return e/np.sum(e)

    def forward(self,x):

        h = np.tanh(
            self.Wxh @ x +
            self.Whh @ self.context +
            self.bh
        )

        y = self.softmax(self.Why @ h + self.by)

        return h,y

    def train_step(self,x,target):

        h,y = self.forward(x)

        dy = y-target

        dWhy = np.outer(dy,h)
        dby = dy

        dh = self.Why.T @ dy
        dh_raw = (1-h**2)*dh

        dWxh = np.outer(dh_raw,x)
        dWhh = np.outer(dh_raw,self.context)
        dbh = dh_raw

        self.Why -= self.lr*dWhy
        self.by  -= self.lr*dby

        self.Wxh -= self.lr*dWxh
        self.Whh -= self.lr*dWhh
        self.bh  -= self.lr*dbh

        self.context = h + self.mu*self.context

        loss = -np.sum(target*np.log(y+1e-12))
        return loss