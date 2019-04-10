import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from costcla.metrics import cost_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, make_scorer, brier_score_loss
from sklearn.model_selection import learning_curve, cross_val_score


def report(results, n_top=3) -> None:
    """
    Utility function to report best scores of Grid/Random Search CV.

    :param results: the search results.
    :param n_top: the number of top scores to show.
    """
    for i in range(1, n_top + 1):
        scores = np.flatnonzero(results['rank_test_score'] == i)
        for score in scores:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][score],
                results['std_test_score'][score]))
            print('Parameters: {0}'.format(results['params'][score]))
            print('')


def estimators_vs_acc(classifier, x_data, y_data, estimators_array) -> None:
    """
    Plot number of estimators vs accuracy +- std.

    :param classifier: the classifier to be used.
    :param x_data: the features.
    :param y_data: the classes.
    :param estimators_array: an array containing the number of estimators to be used for each CV fit.
    """
    # Arrays to store mean and std.
    bg_clf_cv_mean = []
    bg_clf_cv_std = []

    # For each number of estimators run a 10 fold CV and store the results.
    for n_est in estimators_array:
        bagging_clf = BaggingClassifier(base_estimator=classifier,
                                        n_estimators=n_est, random_state=0)
        scores = cross_val_score(bagging_clf, x_data, y_data, cv=10,
                                 scoring='accuracy', verbose=2, n_jobs=-1)
        bg_clf_cv_mean.append(scores.mean())
        bg_clf_cv_std.append(scores.std())

    # Bound upper and lower bounds of the error bar to [0, 1].
    y_min = np.asarray([max(mean - std, 0) for mean, std in zip(bg_clf_cv_mean, bg_clf_cv_std)])
    y_max = np.asarray([min(mean + std, 1) for mean, std in zip(bg_clf_cv_mean, bg_clf_cv_std)])
    y_bot = bg_clf_cv_mean - y_min
    y_top = y_max - bg_clf_cv_mean

    # Plot the accuracy+-std vs number of estimators.
    plt.figure(figsize=(12, 8))
    (_, caps, _) = plt.errorbar(estimators_array, bg_clf_cv_mean,
                                yerr=(y_bot, y_top), c='blue', fmt='-o', capsize=5)

    # Configure the plot.
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.ylabel('Accuracy')
    plt.xlabel('Ensemble Size')
    plt.title('Bagging Tree Ensemble')
    plt.show()


def plot_accuracy_stacking(label_list, clfs, x_data, y_data) -> None:
    """
    Plot accuracy +- std for each classifier used in the stacking model
    and for the stacking classifier.

    :param label_list: a list containing the names of the classifiers for the plot.
    :param clfs: the classifiers.
    :param x_data: the features.
    :param y_data: the classes.
    """
    # Arrays to store mean and std.
    clf_cv_mean = []
    clf_cv_std = []

    # For each classifier run a 10 fold CV and store the results.
    for classifier, label in zip(clfs, label_list):
        scores = cross_val_score(classifier, x_data, y_data, cv=10, scoring='accuracy', n_jobs=-1)
        print("Accuracy: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
        clf_cv_mean.append(scores.mean())
        clf_cv_std.append(scores.std())
        classifier.fit(x_data, y_data)

    # Bound upper and lower bounds of the error bar to [0, 1].
    y_min = np.asarray([max(mean - std, 0) for mean, std in zip(clf_cv_mean, clf_cv_std)])
    y_max = np.asarray([min(mean + std, 1) for mean, std in zip(clf_cv_mean, clf_cv_std)])
    y_bot = clf_cv_mean - y_min
    y_top = y_max - clf_cv_mean

    # Plot the accuracy+-std for each classifier.
    plt.figure(figsize=(12, 8))
    (_, caps, _) = plt.errorbar(range(len(clfs)), clf_cv_mean,
                                yerr=(y_bot, y_top), c='blue', fmt='-o', capsize=5)

    # Configure the plot.
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.xticks(range(len(clfs)), label_list)
    plt.ylabel('Accuracy')
    plt.xlabel('Classifier')
    plt.title('Stacking Ensemble')
    plt.show()


def plot_learning_curve(estimators, names, x_data, y_data) -> None:
    """
    Function to plot the learning curve of the stacking model and all its submodels,

    given some training data.

    :param estimators: an array containing all the 6 estimators.
    :param names: an array containing the estimators names.
    :param x_data: the features.
    :param y_data: the labels.
    """
    f, axes = plt.subplots(2, 3, figsize=(18, 12), sharey='all', squeeze=False)

    for counter, (estimator, ax) in enumerate(zip(estimators, axes.reshape(-1))):
        train_sizes, train_scores, test_scores = learning_curve(estimator, x_data, y_data,
                                                                train_sizes=np.linspace(0.2, 1.0, 5), cv=10, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title(names[counter] + " Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")


def full_report(true, predicted, averaging='macro') -> None:
    """
    Shows a full classification report.

    :param true: the true labels.
    :param predicted: the predicted labels.
    :param averaging: the averaging method to be used.
    """
    print('Final Results')
    print('---------------------')
    print('Accuracy       {:.4f}'
          .format(accuracy_score(true, predicted)))
    print('Precision      {:.4f}'
          .format(precision_score(true, predicted, average=averaging)))
    print('Recall         {:.4f}'
          .format(recall_score(true, predicted, average=averaging)))
    print('F1             {:.4f}'
          .format(f1_score(true, predicted, average=averaging)))


def _cs_report(true, predicted, label_names, cost_matrix) -> None:
    """
    Shows a full cost sensitive classification report.

    :param cost_matrix: the cost matrix.
    :param label_names: the class names.
    :param true: the true labels.
    :param predicted: the predicted labels.
    """
    # Show a classification report.
    print(classification_report(true, predicted, target_names=label_names))

    # Create a confusion matrix with the metrics.
    matrix = confusion_matrix(true, predicted)

    # Create a heatmap of the confusion matrix.
    plt.figure(figsize=(8, 8))
    sns.heatmap(matrix, annot=True, fmt='d', linewidths=.1, cmap='YlGnBu',
                cbar=False, xticklabels=label_names, yticklabels=label_names)
    plt.title('Total Classification Cost -> {}'.format(cost_loss(true, predicted, cost_matrix)), fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.xlabel('True output', fontsize='x-large')
    plt.ylabel('Predicted output', fontsize='x-large')
    plt.savefig(fname='confusion_matrix.png')
    plt.show()


def full_cs_report(y_test, y_forest, y_svm, y_bayes, label_names, cost_matrix) -> None:
    """
    Make a report for all the cost sensitive classifiers.

    :param y_test: the test labels.
    :param y_forest: the random forest predicted labels.
    :param y_svm: the svm predicted labels.
    :param y_bayes: the bayes predicted labels.
    :param cost_matrix: the cost matrix.
    :param label_names: the class names.
    """
    print('Random Forest: \n')
    _cs_report(y_test, y_forest, label_names, cost_matrix)
    print('\n---------------------------------------------------------------\n')
    print('SVM: \n')
    _cs_report(y_test, y_svm, label_names, cost_matrix)
    print('\n---------------------------------------------------------------\n')
    print('Bayes: \n')
    _cs_report(y_test, y_bayes, label_names, cost_matrix)


def cost_loss_func(y_true, y_pred) -> int:
    """
    Define a cost loss function.

    :param y_true: the true labels.
    :param y_pred: the predicted labels.
    :return: the total cost.
    """
    if y_true.shape[0] is not y_pred.shape[0]:
        raise ValueError('True labels and predicted labels shapes do not match!')

    total_cost = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            total_cost += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            total_cost += 5

    return total_cost


# Define cost loss scorer.
cost = make_scorer(cost_loss_func, greater_is_better=False)


def plot_calibration_curve(est, name, X_train, y_train, X_test, y_test) -> None:
    """
    Plot calibration curve for an estimator without and with calibration.

    :param est: the estimator.
    :param name: the estimator's name.
    :param X_train: the train data.
    :param y_train: the train labels.
    :param X_test: the test data.
    :param y_test: the test labels.
    """
    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=10, method='sigmoid')

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name), (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('{}\nCalibration plots  (reliability curve)'.format(name))

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
