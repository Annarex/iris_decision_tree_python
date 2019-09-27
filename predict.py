import googleapiclient.discovery

#gcloud ai-platform predict --model=scikit_iris_randomforest_temp --version=v1 --json-instances=input.json


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([[float]]): List of input instances, where each input
           instance is a list of floats.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    print(response['predictions'])
    return response['predictions']


instances = [[6.8,  2.8,  4.8,  1.4],[6.0,  3.4,  4.5,  1.6]]

prediction = predict_json('scikit-iris','scikit_iris_randomforest_temp', instances, 'v1')
