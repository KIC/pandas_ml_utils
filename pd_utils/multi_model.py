
class MultiModel(object):

    def __init__(self):
        # filename
        # data_provider: Callable[[], pd.Dataframe]
        # feature_provider: Callable[[pd.Dataframe], pd.Dataframe]
        # feature_provider_argument_space: Dict[str, Dict[str, Any]]
        pass

    def fit(self):
        # pass in a model provider to fit and then save all the models
        pass

    def predict(self):
        # return a prediction of every model
        pass