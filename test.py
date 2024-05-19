from auto_ml_flow.client.v1.api.datasets import DatasetsClient
from auto_ml_flow.client.v1.models.datasets import CreateDatasetPayload

if __name__ == "__main__":
    ds_client = DatasetsClient("http://127.0.0.1:8000/")
    with open(
        "/Users/maxim/Documents/Other/Education/diplom/auto_ml_flow/auto-ml-flow/ttt.txt"
    ) as f:
        ds_client.create(CreateDatasetPayload(n_samples=4, n_features=4, run=2), f)
