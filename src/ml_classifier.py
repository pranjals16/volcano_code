import numpy as np
import re

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier, LassoCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC

BASE_PATH = "../data/"


def remove_hyperlink(text):
    text = re.sub(r"http\S+", '',  text.decode("utf-8"))
    return text


def read_files(file_type):
    x_list = []
    y_list = []
    with open(BASE_PATH + file_type + "_x.txt", "rb") as f:
        for line in f:
            processed_line = remove_hyperlink(line)
            x_list.append(processed_line.strip())

    with open(BASE_PATH + file_type + "_y.txt", "rb") as f:
        for line in f:
            y_list.append(int(line.strip()))

    return x_list, y_list


def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def rf_random_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 30, 50, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(x_train_tf_idf, y_train)
    print(rf_random.best_params_)
    # {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 788, 'max_features': 'sqrt', 'min_samples_split': 20
    # #'max_depth': 110}

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, x_test_tf_idf, y_test)
    print(random_accuracy)


def rf_grid_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test):
    param_grid = {
        'bootstrap': [True],
        'max_depth': [100],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': [15, 20, 25],
        'n_estimators': [600, 700, 800]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train_tf_idf, y_train)
    print("Best Parameters of Grid Search: ", grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    # Best Parameters of Grid Search:  {'max_features': 'sqrt', 'min_samples_split': 15, 'bootstrap': True,
    # 'n_estimators': 700, 'max_depth': 100}
    print("Random Forest Grid Search: ", evaluate(best_grid, x_test_tf_idf, y_test))


def mnb_grid_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test):
    param_grid = {
        'fit_prior': [True, False],
        'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    }
    mnb = MultinomialNB()
    grid_search = GridSearchCV(estimator=mnb, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train_tf_idf, y_train)
    print("Best Parameters of Grid Search: ", grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    # Best Parameters of Grid Search:  {'alpha': 0.1, 'fit_prior': False}
    print("Multinomial Naive Bayes Grid Search: ", evaluate(best_grid, x_test_tf_idf, y_test))


def plot_random(test_features, y_true):
    mnb_clf = joblib.load('../model/mnb.pkl')
    rf_clf = joblib.load('../model/rf.pkl')
    adb_clf = joblib.load('../model/adb.pkl')
    svc_linear_clf = joblib.load('../model/svc_linear.pkl')
    svc_linera_lda_clf = joblib.load('../model/svc_lda.pkl')
    y_proba_mnb = mnb_clf.predict_proba(test_features)[:, 1]
    y_proba_rf = rf_clf.predict_proba(test_features)[:, 1]
    y_proba_adb = adb_clf.predict_proba(test_features)[:, 1]
    y_proba_svc_linear = svc_linear_clf.predict_proba(test_features)[:, 1]
    y_proba_svc_linear_feat_select = svc_linera_lda_clf.predict_proba(test_features_red)[:, 1]
    precision1, recall1, _ = precision_recall_curve(y_true, y_proba_mnb)
    precision2, recall2, _ = precision_recall_curve(y_true, y_proba_rf)
    precision3, recall3, _ = precision_recall_curve(y_true, y_proba_adb)
    precision4, recall4, _ = precision_recall_curve(y_true, y_proba_svc_linear)
    precision5, recall5, _ = precision_recall_curve(y_true, y_proba_svc_linear_feat_select)
    plt.step(recall1, precision1, alpha=0.6, where='post', label="MNB")
    plt.step(recall2, precision2, alpha=0.6, where='post', label="RF")
    plt.step(recall3, precision3, alpha=0.6, where='post', label="ADB")
    plt.step(recall4, precision4, alpha=0.6, where='post', label="SVC_Linear")
    plt.step(recall5, precision5, alpha=0.6, where='post', label="SVC_Linearv2")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.legend()
    plt.title('2-class Precision-Recall curve')
    plt.show()


def plot_pr_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.step(recall, precision, color='r', alpha=0.2,
             where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.show()


def evaluate(model, test_features, y_true):
    y_pred = model.predict(test_features)
    y_proba = model.predict_proba(test_features)[:, 1]
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    auc_score = metrics.roc_auc_score(y_true, y_proba)
    precision_score = metrics.average_precision_score(y_true, y_proba)
    plot_pr_curve(y_true, y_proba)
    return accuracy, auc_score, precision_score


def main():
    x_train, y_train = read_files("train")
    print("Training File Read - Size: ", len(x_train))
    x_test, y_test = read_files("test")
    print("Test File Read - Size: ", len(x_test))
    x_val, y_val = read_files("val")
    print("Validation File Read - Size: ", len(x_val))

    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)
    x_test_counts = count_vect.transform(x_test)

    # Tf-Idf Transformation
    tf_idf_transformer = TfidfTransformer()
    x_train_tf_idf = tf_idf_transformer.fit_transform(x_train_counts)
    x_test_tf_idf = tf_idf_transformer.transform(x_test_counts)

    plot_random(x_test_tf_idf, y_test)
    '''
    svm_clf = SVC(probability=True, kernel='linear').fit(x_train_tf_idf, y_train)
    accuracy, auc_score, precision_score = evaluate(svm_clf, x_test_tf_idf, y_test)
    print("Linear SVM Metrics: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}"
          .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))
    joblib.dump(svm_clf, '../model/svc_linear.pkl')

    clf_nb = MultinomialNB(alpha=0.1, fit_prior=False).fit(x_train_tf_idf, y_train)
    accuracy, auc_score, precision_score = evaluate(clf_nb, x_test_tf_idf, y_test)
    print("Multinomial Naive Bayes Metrics: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}"
          .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))
    joblib.dump(clf_nb, '../model/mnb.pkl')

    clf_rf = RandomForestClassifier(max_features='sqrt', min_samples_split=15, bootstrap=True,
                                    n_estimators=1000, max_depth=200).fit(x_train_tf_idf, y_train)
    accuracy, auc_score, precision_score = evaluate(clf_rf, x_test_tf_idf, y_test)
    print("Random Forest Metrics: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}"
          .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))
    joblib.dump(clf_rf, '../model/rf.pkl')
    # rf_random_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test)
    # rf_grid_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test)
    # mnb_grid_search(x_train_tf_idf, y_train, x_test_tf_idf, y_test)
    adb_clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.2).fit(x_train_tf_idf, y_train)
    accuracy, auc_score, precision_score = evaluate(adb_clf, x_test_tf_idf, y_test)
    print("AdaBoost Metrics: Accuracy-{accuracy}, AUC-{auc_score} and PR_Score-{precision_score}"
          .format(accuracy=accuracy, auc_score=auc_score, precision_score=precision_score))
    joblib.dump(adb_clf, '../model/adb.pkl')
    '''


if __name__ == "__main__":
    main()
