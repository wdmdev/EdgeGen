from huggingface_hub import login
from huggingface_hub import HfApi

class HuggingFaceDownloader():

    def __init__(self):
        self.base_url = "https://huggingface.co/"
        self.api = HfApi()

        try:
            self.api.whoami()
        except:
            login()





if __name__ == "__main__":
    # This will prompt you to enter your Hugging Face token
    # login()

    base_url = "https://huggingface.co/"

    # Initialize the Hugging Face API client
    api = HfApi()

    # Retrieve models with a specific task, e.g., image classification
    models = api.list_models(filter="image-classification", fetch_config=True)

    # Filter models that are convolutional
    model_infos = {}
    for model in models:
        model_info = {}

        model_id = model.modelId
        model_info["model_id"] = model_id

        model_url = base_url + model_id

        if model.config:
            model_info["config"] = model.config
        else:
            model_info["config"] = None

        # Retrieve the model's metadata
        model_metadata = api.model_info(model_id, files_metadata=True)
        model_info["metadata"] = model_metadata

    model_count = len(model_infos)
    models_without_config = sum([1 for model_info in model_infos.values() if model_info["config"] is None])
    print(f"Total models: {model_count}")
    print(f"Models without config: {models_without_config}")