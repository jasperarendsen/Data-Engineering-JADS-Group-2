# MLP for Pima Indians Dataset saved to single file
# see https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import json
import os

from sklearn import ensemble
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from pickle


# def train(dataset):
#     # split into input (X) and output (Y) variables
#     X = dataset[:, 0:8]
#     Y = dataset[:, 8]
#     # define model
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # Fit the model
#     model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X, Y, verbose=0)
#     print(model.metrics_names)
#     text_out = {
#         "accuracy:": scores[1],
#         "loss": scores[0],
#     }
#     # Saving model in a given location (provided as an env. variable
#     model_repo = os.environ['MODEL_REPO']
#     if model_repo:
#         file_path = os.path.join(model_repo, "model.h5")
#         model.save(file_path)
#     else:
#         model.save("model.h5")
#         return json.dumps({'message': 'The model was saved locally.'},
#                           sort_keys=False, indent=4), 200

#     print("Saved the model to disk")
#     return json.dumps(text_out, sort_keys=False, indent=4)


def train(dataset): 
    X = dataset[dataset.columns[3:]]
    Y = dataset[dataset.columns[2]]
    
    #Clustering the data
    X_cluster = X[['setting1', 'setting2', 'setting3']]

    #creates the clusters
    kmeans = KMeans(n_clusters=3).fit(X_cluster)
    X['settings_clusters'] = kmeans.predict(X_cluster)
    
    features = dataset.columns[3:-1]
    for feature in features:
        #Creating min, max and delta variables
        X['max_' + feature] = dataset.groupby('engine_id')[feature].cummax()
        X['min_' + feature] = dataset.groupby('engine_id')[feature].cummin()

        X['delta_' + feature] = dataset.groupby('engine_id')[feature].diff()
        X['delta_' + feature].fillna(0, inplace=True)
    
    ###oude code
    params = {'n_estimators': 400, 'max_depth': 5, 'learning_rate': 0.25}
    
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(X, Y)

    
    # scores = model.evaluate(X, Y, verbose=0)
    text_out = {
        "R2:": model.score(X,Y)
    }
    # Saving model in a given location (provided as an env. variable
    model_repo = os.environ['MODEL_REPO']
    if model_repo:
        file_path = os.path.join(model_repo, "model.pkl")
        pickle.dump(model, open(file_path, 'wb'))
    else:
        pickle.dump(model, open('model.pkl', 'wb'))
        return json.dumps({'message': 'The model was saved locally.'},
                          sort_keys=False, indent=4), 200

    print("Saved the model to disk")
    return json.dumps(text_out, sort_keys=False, indent=4)