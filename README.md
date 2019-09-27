**This is a simple hello world style decision tree written in python that works with the iris data set**

**Installation:**

https://cloud.google.com/ml-engine/docs/scikit/using-pipelines

Install Google-Cloud SDK
-------------------------
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk


gcloud init

gcloud projects list
gcloud projects delete <ProjectID>


gcloud components update

sudo apt-get update && sudo apt-get --only-upgrade install kubectl google-cloud-sdk google-cloud-sdk-app-engine-grpc google-cloud-sdk-pubsub-emulator google-cloud-sdk-app-engine-go google-cloud-sdk-firestore-emulator google-cloud-sdk-cloud-build-local google-cloud-sdk-datastore-emulator google-cloud-sdk-app-engine-python google-cloud-sdk-cbt google-cloud-sdk-bigtable-emulator google-cloud-sdk-app-engine-python-extras google-cloud-sdk-datalab google-cloud-sdk-app-engine-java

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
echo $BUCKET_NAME
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil cp ./model.joblib gs://$BUCKET_NAME/model.joblib

gsutil ls
gsutil ls gs://scikit-iris-mlengine/



sudo apt install python-pip
pip install scipy
pip install sklearn
pip install tensorflow
python iris.py

pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
python predict.py


gcloud ai-platform models list
gcloud ai-platform models create "[YOUR-MODEL-NAME]"


MODEL_DIR="gs://your_bucket_name/"
VERSION_NAME="[YOUR-VERSION-NAME]"
MODEL_NAME="[YOUR-MODEL-NAME]"
FRAMEWORK="[YOUR-FRAMEWORK_NAME]"
Create the version:

gcloud ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --origin $MODEL_DIR \
  --runtime-version=1.14 \
  --framework $FRAMEWORK \
  --python-version=3.5

Creating the version takes a few minutes. When it is ready, you should see the following output:

Creating version (this might take a few minutes)......done.
Get information about your new version:

gcloud ai-platform versions describe $VERSION_NAME \
  --model $MODEL_NAME


gcloud ai-platform local predict --model-dir=. --json-instances=input.json --framework=SCIKIT_LEARN
gcloud ai-platform predict --model=scikit_iris_randomforest_temp --version=v1 --json-instances=input.json
