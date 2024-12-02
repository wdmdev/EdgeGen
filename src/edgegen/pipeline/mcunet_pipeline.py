from .pipeline import Pipeline

class McunetPipeline(Pipeline):
    def __init__(self, archGenerator, evalEngine, modelRepo, logger):
        super().__init__(archGenerator, evalEngine, modelRepo, logger)

    def _configure_logger(self):
        pass

    def run(self):
        architecture = self.archGenerator.generate()
        eval_result = self.evalEngine.evaluate(architecture)
        