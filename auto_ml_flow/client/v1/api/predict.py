from auto_ml_flow.client.base import BaseClient
from auto_ml_flow.client.v1.models.predict import MetaAlgoFeatures, MetaAlgoPredictions


class MetaAlgoClient(BaseClient):
    DEFAULT_PREFIX = "/api/v1/meta-algos"

    def predict(self, param: MetaAlgoFeatures) -> MetaAlgoPredictions:
        return self._post(
            f"{self.DEFAULT_PREFIX}/predict/", data=param.model_dump(), model=MetaAlgoPredictions
        )
