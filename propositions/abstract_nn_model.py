class AbstractNNModel:
    def __init__(self, **params):
        pass

    def validate_parameters(self, params):
        raise NotImplementedError

    def init_model(self, params):
        raise NotImplementedError
