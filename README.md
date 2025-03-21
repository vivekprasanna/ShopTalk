# ShopTalk

```aidl
pip install -r requirements.txt
```

## Run main.py in the root directory as a python file.
```
python main.py
```

# Finally run the following command
```
python app.py
```

This should get the swagger accessible at 
```http://127.0.0.1:8080/```

## To run streamlit 
```
streamlit run /Users/vivekprabu/PycharmProjects/ShopTalk/streamlit-ui.py
```

## To run E2E test
```commandline
pip install -r requirements-test.txt
pytest -m e2e
```

## MLFlow experiment link
https://dagshub.com/vivekprasanna.prabhu/ShopTalk.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

You have to export the environment variables for running the evaluation/training part of the project
ENV MLFLOW_TRACKING_USERNAME
ENV MLFLOW_TRACKING_PASSWORD
ENV MLFLOW_TRACKING_URI=https://dagshub.com/vivekprasanna.prabhu/ShopTalk.mlflow

This is already part of docker image.
You can run the docker image using the following command - this will expose the right ports
```commandline
docker run -p 8080:8080 -p 8501:8501 vivekprabu/shop-talk-app:latest
```
