import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#####################Overview######################################
# In this file we add the function to learn the system model periodically.
# However, we have the plan to add the increamental learning of the system mdoel soon.
# Moreover, our experience shows that with type of system we have, random forest performs best in terms of both accuracy and training time.
# Nevertheless, it can be easily replaced with other regressors.

####################### Input arguments ###########################
args = sys.argv

number_of_training_iteration = args[1]
data_path = args[2]

if len(args)>3:
    target_path = args[3]

##################### Evaluation and training functions ##########################
def test_train_nmae_r2score(test_predicted_values, train_predicted_values, test_set, train_set):
    col_size = train_set.shape[1]
    test_nmaes = []
    train_names = []
    test_r2s = []
    train_r2s = []
    for i in range(col_size):
        diff_test = np.abs(test_predicted_values[:,i] - test_set.iloc[:,i])
        test_nmae = (diff_test.mean()) / test_set.iloc[:,i].mean()
        test_r2score = r2_score(test_set.iloc[:,i], test_predicted_values[:,i])
        diff_train = np.abs(train_predicted_values[:,i] - train_set.iloc[:,i])
        train_nmae = (diff_train.mean()) / train_set.iloc[:,i].mean()
        train_r2score = r2_score(train_set.iloc[:,i], train_predicted_values[:,i])
        test_nmaes.append(test_nmae)
        train_names.append(train_nmae)
        test_r2s.append(test_r2score)
        train_r2s.append(train_r2score)
        print("avg diff: ", diff_test.mean())
    return test_nmaes, train_names, test_r2s, train_r2s

def offline_training(data_path, x_features, y_features, iteration, test_size=0.05, n_estimators=120, target_path="./"):

    '''
    This function is used for the offline learning of the system model using the collected data
    :param data_path: Path the file that data is located.
    :param x_features: The feature set for the design matrix
    :param y_features: The labels for the target
    :return: The accuracy of the model in terms of NMAE and R2
    '''
    data = pd.read_csv(data_path)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Splitting the data set into train and test sets
    X = data[x_features]
    Y = data[y_features]
    merge_raw = pd.concat([X, Y], axis=1)
    train, test = train_test_split(merge_raw, test_size=test_size)
    train = merge_raw

    # Sort index in train, test
    train = train.sort_index(axis=0)
    test = test.sort_index(axis=0)

    # Preparing the train and test set
    X_train = train[x_features].iloc[1:]
    X_test = test[x_features].iloc[1:]
    Y_train = train[y_features].iloc[1:]
    Y_test = test[y_features].iloc[1:]

    # Training the random forest regressor
    regressor = RandomForestRegressor(n_estimators=n_estimators)
    regressor.fit(X_train.values, Y_train)
    pred_test = regressor.predict(X_test)
    predict_test = (np.array(pred_test)).reshape(pred_test.shape[0], Y_test.shape[1])
    pred_train = regressor.predict(X_train)
    predict_train = (np.array(pred_train)).reshape(pred_train.shape[0], Y_test.shape[1])

    # Evalauting the learned model
    test_nmae, train_nmae, test_r2score, train_r2score = test_train_nmae_r2score(predict_test, predict_train, Y_test,
                                                                                 Y_train)

    # Saving the learned model in the given path
    joblib.dump(regressor, target_path + "system_model" + str(iteration)+".joblib")

    return test_nmae, train_nmae, test_r2score, train_r2score

if __name__ == "__main__":
    x_features = ["c1", "p11", "l1", "cl1"]
    y_features = ["d1"]
    # In these iterations the regressor is updated reading the updated data files.
    for i in range(number_of_training_iteration):
        test_nmae, train_nmae, test_r2score, train_r2score = offline_training(data_path, x_features, y_features, i)