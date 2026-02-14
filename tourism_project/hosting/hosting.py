
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_project/deployment",                   # the local folder containing our deployment files
    repo_id="vallabbharath/Tourism-Prediction-MLOps-Project",   # the existing space created as pre-req
    repo_type="space",                                          # dataset, model, or space
    path_in_repo=""                                             # optional: subfolder path inside the repo
)

print("Deployment files uploaded to existing Space.")
