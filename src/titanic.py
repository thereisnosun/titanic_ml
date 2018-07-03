import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score, cross_val_predict

DATA_PATH = '../data/'


def load_dataset(dataset_path):
    csv_path = os.path.join(DATA_PATH, dataset_path)
    return pd.read_csv(csv_path)


def get_feature_matrix(dataset):
    dataset['Sex'] = dataset['Sex'].map(lambda x: 1 if x=='male' else 0)
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean());
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean());

    dataset['Embarked'] = dataset['Embarked'].map(lambda x: 0 if x == 'C' else 1 if x == 'Q' else 2)
    #dataset['Age'] = dataset['Age'].fillna(0);
    feature_matrix = dataset.as_matrix(columns=['Pclass', 'Embarked', 'Fare', 'Sex', 'Age'])
    print(dataset.isnull().any())
    return feature_matrix

def create_prediction_file(test_dataset, prediction):
    df = pd.DataFrame({'PassengerId':test_dataset['PassengerId'], 'Survived':prediction})
    df.to_csv('../data/predict_res.csv', index=False)


def test_prediction(random_forest):
    test_dataset = load_dataset('test.csv')
    feature_matix = get_feature_matrix(test_dataset)
    print ("Prediction on unseen data")
    predict_result = random_forest.predict(feature_matix)
    create_prediction_file(test_dataset, predict_result)



#probably the most correlation is due to passanger class, sex and age
def main():
    train_dataset = load_dataset('train.csv')
    print(list(train_dataset), len(train_dataset))
    #print(train_dataset['PassengerId'])
    # plt.scatter(train_dataset["Age"], train_dataset["Fare"], c=train_dataset["Survived"]);
    # plt.show();
    feature_matrix = get_feature_matrix(train_dataset)
    print(feature_matrix)
    feature_output = train_dataset.as_matrix(columns=['Survived'])

    print(feature_matrix.shape)
    # rbf_feature = RBFSampler(random_state=1, gamma=1)
    # feature_matrix = rbf_feature.fit_transform(feature_matrix)
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(feature_matrix, feature_output)

    val_score = cross_val_score(sgd_clf, feature_matrix, feature_output, cv=3, scoring="accuracy");
    print(val_score)

    survive_train_predict = cross_val_predict(sgd_clf, feature_matrix, feature_output, cv=3)
    #print(survive_train_predict)
    #print(sgd_clf.predict([feature_matrix[0]]), sgd_clf.predict([feature_matrix[1]]), sgd_clf.predict([feature_matrix[3]]))

    random_forest = RandomForestClassifier(random_state=42)
    print (random_forest.fit(feature_matrix, feature_output))
    val_score = cross_val_score(random_forest, feature_matrix, feature_output, cv=3, scoring="accuracy");
    print(val_score)
    survive_prob = cross_val_predict(random_forest, feature_matrix, feature_output, cv=3)
    #print(survive_prob)

    test_prediction(random_forest)
    #TODO: probably some additional feature analysis or engineering


if __name__ == "__main__":
    main()
