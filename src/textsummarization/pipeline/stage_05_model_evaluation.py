from textsummarization.config.configuration import ConfigurationManager
from textsummarization.conponents.model_training import ModelTrainer
from textsummarization.conponents.model_evaluation import ModelEvaluation

class DataModelEvaluationPipeline:
    def __init__(self):
        pass



    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()