class ParameterNotAllowedException(Exception):
    def __init__(self, parameter_name):
         super().__init__(f"Parameter not allowed: {parameter_name}")

class MissingParameterException(Exception):
    def __init__(self, parameter_name):
         super().__init__(f"Parameter missing: {parameter_name}")

class WrongParameterTypeException(Exception):
     def __init__(self, parameter, value_type, expected_type) -> None:
          super().__init__(f"Wrong value type for parameter {parameter}: Expected {expected_type}, received {value_type}")