import time
from locust import HttpUser, task, between, SequentialTaskSet
import json

with open('../test_resourses/training-db/train_data.json') as outfile:
    training_data = json.load(outfile)

with open('../test_resourses/training-db/predict_data.json') as outfile:
    predict_data = json.load(outfile)


class QuickstartUser(SequentialTaskSet):
    @task
    def load_data(self):
        self.client.post("http://35.232.3.223:5000/training-db/rul", json={
            "columns": [
                "engine_id",
                "cycle",
                "RUL",
                "setting1",
                "setting2",
                "setting3",
                "s1",
                "s2",
                "s3",
                "s4",
                "s5",
                "s6",
                "s7",
                "s8",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s18",
                "s19",
                "s20",
                "s21"
            ]
        })
        self.client.put("http://35.232.3.223:5000/training-db/rul", json=training_data)

    @task
    def train_model(self):
        self.client.post("http://35.223.182.124:5001/training-cp/mlp")

    @task
    def predict(self):
        self.client.post("http://34.67.131.42:5002/prediction-cp/mlp", json=predict_data)

class LoggedInUser(HttpUser):
    wait_time = between(1, 2)
    tasks = {QuickstartUser:2}