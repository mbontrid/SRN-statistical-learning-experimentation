import numpy as np

from exceptions.model_exceptions import MissingParameterException, ParameterNotAllowedException, WrongParameterTypeException
from models.abstract_nn_model import AbstractNNModel

class SRN(AbstractNNModel):

    def __init__(self, params):
        self.allowed_parameters = {'vocab_size': int,'hidden_size': int, 'lr': float, 'mu': float, 'clearval': float, 'epochs': int}
        self.init_model(params)

    def validate_parameters(self, params):
        for parameter, value in params.items():
            if parameter not in self.allowed_parameters.keys():
                raise ParameterNotAllowedException(parameter)
            expected_type = self.allowed_parameters[parameter]
            if not isinstance(value, expected_type):
                raise WrongParameterTypeException(parameter, value.type(), expected_type)
        for parameter in self.allowed_parameters:
            if parameter not in params.key():
                raise MissingParameterException(parameter)
        return True

        

    def init_model(self, params):

        self.validate_parameters(params);

        
        self.input_size = params['vocab_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['vocab_size']

        self.lr = params['lr']
        self.mu = params['mu']
        self.clearval = params['clearval']

        self.Wxh = np.random.uniform(-0.1,0.1,(self.hidden_size,self.output_size))
        self.Whh = np.random.uniform(-0.1,0.1,(self.hidden_size,self.hidden_size))
        self.Why = np.random.uniform(-0.1,0.1,(self.output_size,self.hidden_size))

        self.bh = np.zeros(self.hidden_size)
        self.by = np.zeros(self.output_size)

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
