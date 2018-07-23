import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import string
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import CategoricalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

DATA_PATH = '../data/'


def load_dataset(dataset_path):
    csv_path = os.path.join(DATA_PATH, dataset_path)
    return pd.read_csv(csv_path)


def substrings_in_string(big_string, substrings):
    for curr_str in substrings:
        if big_string.find(curr_str) != -1:
            return curr_str
    return np.nan


def encode_categories(dataset, column_name):
    dataset = pd.concat([dataset, pd.get_dummies(dataset['Title'], prefix='Title')], axis=1)
    print(dataset.head())
    return dataset



def get_tittle(dataset):
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    dataset['Title'] = dataset['Name'].map(lambda x: substrings_in_string(x, title_list))
    def replace_titles(x):
        title=x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Master']:
            return 'Officer'
        elif title in ['Countess', 'Mme', 'Mrs', 'Miss', 'Mlle', 'Ms']:
            return 'Mrs'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
    dataset['Title'] = dataset.apply(replace_titles, axis=1)

    return dataset['Title']


#TODO: fare split on members number
def get_feature_matrix(dataset):
    dataset['Title'] = get_tittle(dataset)
    dataset = encode_categories(dataset, 'Title')
    dataset['Sex'] = dataset['Sex'].map(lambda x: 1 if x=='female' else 0)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean());
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean());

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['ActualFare'] = dataset['Fare'] / (dataset['FamilySize'])
    dataset['ActualFare'] = dataset['ActualFare'].fillna(dataset['ActualFare'].mean())
    dataset['AgeCategory'] = dataset['Age'].map(lambda x: 0 if x <= 16 else 2 if x>=64 else 1)

    print("COUNT - ", dataset['FamilySize'].value_counts(ascending=True))
    #print("COUNT - ", dataset['ActualFare'].value_counts(ascending=True))

    dataset = pd.concat([dataset, pd.get_dummies(dataset['Embarked'], prefix='Embarked')], axis=1)
    print(list(dataset))
    feature_matrix = dataset.as_matrix(columns=['Fare', 'Pclass', 'Sex',  'Age',   'FamilySize',
                                                 'Title_Officer', 'Title_Mrs', 'Title_Mr',
                                                 'Embarked_Q', 'Embarked_S', 'Embarked_C', 'AgeCategory'])
    print(feature_matrix)
    return feature_matrix


def create_prediction_file(test_dataset, prediction):
    df = pd.DataFrame({'PassengerId':test_dataset['PassengerId'], 'Survived':prediction})
    df.to_csv('../data/predict_res.csv', index=False)


def test_prediction(random_forest):
    test_dataset = load_dataset('test.csv')
    feature_matix = get_feature_matrix(test_dataset)
    print ("Prediction on unseen data")
    # sc = StandardScaler()
    # sc.fit_transform(feature_matix)
    predict_result = random_forest.predict(feature_matix)
    create_prediction_file(test_dataset, predict_result)


def plot_histogram(x, y):
    x.dropna(inplace=True)
    y.dropna(inplace=True)
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    n_bins=20
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(x, bins=n_bins)
    axs[1].hist(y, bins=n_bins)
    plt.show()

def hyper_param_search(feature_matrix, feature_output):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10, 11]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestClassifier(random_state=42,  n_estimators=500, n_jobs=-1);
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(feature_matrix, feature_output)
    print(grid_search.best_params_)
    return grid_search


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def compute_scores(model, feature_matrix, feature_output):
    scores = cross_val_score(model, feature_matrix, feature_output, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)


#probably the most correlation is due to passanger class, sex and age
def main():
    train_dataset = load_dataset('train.csv')
    print(list(train_dataset), len(train_dataset))
    #print(train_dataset['PassengerId'])
    # plt.scatter(train_dataset["Age"], train_dataset["Fare"], c=train_dataset["Survived"]);
    # plt.show();

    #plt.bar(train_dataset['Age'])
    # plt.hist([train_dataset['Age'], train_dataset['Survived']])
    # plt.show()


    feature_matrix = get_feature_matrix(train_dataset)
    # print(feature_matrix)
    # plot_histogram(train_dataset['Age'], train_dataset['Survived'])
    feature_output = train_dataset.as_matrix(columns=['Survived'])

     #print(feature_matrix)
    #class_weights = {0:3, 1:1, 2:1, 3:3, 4:2}
    #class_weights = {0: 3, 1: 1}
    # grid_search = hyper_param_search(feature_matrix, feature_output)
    # final_model = grid_search.best_estimator_
    #final_model = RandomForestClassifier(random_state=42, n_estimators=1000,max_features=10)
    random_forest = RandomForestClassifier(random_state=42, n_estimators=1000, max_features=10)
    extra_clf = ExtraTreesClassifier(random_state=13, n_estimators=500, min_samples_split=1.0, max_depth=None)
    log_clf = LogisticRegression()

    final_model = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', random_forest), ('extr', extra_clf)],
        voting='hard')

    final_model.fit(feature_matrix, feature_output)
    #print(list(zip(, final_model.feature_importances_)))
    #print(final_model.feature_importances_)
    val_score = cross_val_score(final_model, feature_matrix, feature_output, cv=3, scoring="accuracy").mean();
    print(val_score)
    #print(final_model.score(feature_matrix, feature_output))
    compute_scores(final_model, feature_matrix, feature_output)
    #survive_prob = cross_val_predict(final_model, feature_matrix, feature_output, cv=3)
    #print(survive_prob)

    test_prediction(final_model)
    #TODO: add cross-validation


if __name__ == "__main__":
    main()
