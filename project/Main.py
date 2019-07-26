from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import BreastCancer as bc
from project import HouseVotes as votes
from project import NaiveBayes as nb
from project import FiveFold as ff
from project import LogisticRegression as lr
import sys

"""
This is the main driver class for Assignment#5.  @author: Tyler Sefcik
"""


# NB and LR for Iris
def run_iris(filename, target_class, class_wanted, iris_names):
    # Setup data
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                         iris_names=iris_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    iris1, iris2, iris3, iris4, iris5 = five_fold.five_fold_sort_class(data=iris_data, sortby=target_class)

    iris_nb1 = nb_iris(iris_data=iris1, target_class=target_class)
    iris_nb2 = nb_iris(iris_data=iris1, target_class=target_class)
    iris_nb3 = nb_iris(iris_data=iris1, target_class=target_class)
    iris_nb4 = nb_iris(iris_data=iris1, target_class=target_class)
    iris_nb5 = nb_iris(iris_data=iris1, target_class=target_class)

    nb_perf = [iris_nb1, iris_nb2, iris_nb3, iris_nb4, iris_nb5]

    iris_lr1 = perform_lr(iris1)
    iris_lr2 = perform_lr(iris2)
    iris_lr3 = perform_lr(iris3)
    iris_lr4 = perform_lr(iris4)
    iris_lr5 = perform_lr(iris5)

    lr_perf = [iris_lr1, iris_lr2, iris_lr3, iris_lr4, iris_lr5]

    return nb_perf, lr_perf


# Run Naive Bayes on Iris
def nb_iris(iris_data, target_class):
    # Split the data set into 2/3 and 1/3
    iris_data_train = iris_data.sample(frac=.667)
    iris_data_test = iris_data.drop(iris_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(iris_data_train.iloc[:, 0:4], iris_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(iris_data_test.iloc[:, 0:4], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=iris_data_test[target_class])

    return nb_perf


# NB and LG for Glass
def run_glass(filename, target_class, class_wanted, glass_names):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            glass_names=glass_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    glass1, glass2, glass3, glass4, glass5 = five_fold.five_fold_sort_class(data=glass_data, sortby=target_class)

    glass_nb1 = nb_glass(glass_data=glass1, target_class=target_class)
    glass_nb2 = nb_glass(glass_data=glass1, target_class=target_class)
    glass_nb3 = nb_glass(glass_data=glass1, target_class=target_class)
    glass_nb4 = nb_glass(glass_data=glass1, target_class=target_class)
    glass_nb5 = nb_glass(glass_data=glass1, target_class=target_class)

    nb_perf = [glass_nb1, glass_nb2, glass_nb3, glass_nb4, glass_nb5]

    glass_lr1 = perform_lr(glass1)
    glass_lr2 = perform_lr(glass2)
    glass_lr3 = perform_lr(glass3)
    glass_lr4 = perform_lr(glass4)
    glass_lr5 = perform_lr(glass5)

    lr_perf = [glass_lr1, glass_lr2, glass_lr3, glass_lr4, glass_lr5]

    return nb_perf, lr_perf


# Run Naive Bayes on Glass
def nb_glass(glass_data, target_class):
    # Split the data set into 2/3 and 1/3
    glass_data_train = glass_data.sample(frac=.667)
    glass_data_test = glass_data.drop(glass_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(glass_data_train.iloc[:, 0:9], glass_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(glass_data_test.iloc[:, 0:9], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=glass_data_test[target_class])

    return nb_perf


# NB and LG for Spambase
def run_spambase(filename, target_class):
    # Setup data
    spambase_obj = spambase.Spambase()
    spambase_data = spambase_obj.setup_data_spambase(filename=filename, target_class=target_class)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    spambase1, spambase2, spambase3, spambase4, spambase5 = five_fold.five_fold_sort_class(data=spambase_data, sortby=target_class)

    spambase_nb1 = nb_spambase(spambase_data=spambase1, target_class=target_class)
    spambase_nb2 = nb_spambase(spambase_data=spambase1, target_class=target_class)
    spambase_nb3 = nb_spambase(spambase_data=spambase1, target_class=target_class)
    spambase_nb4 = nb_spambase(spambase_data=spambase1, target_class=target_class)
    spambase_nb5 = nb_spambase(spambase_data=spambase1, target_class=target_class)

    nb_perf = [spambase_nb1, spambase_nb2, spambase_nb3, spambase_nb4, spambase_nb5]

    spambase_lr1 = perform_lr(spambase1)
    spambase_lr2 = perform_lr(spambase2)
    spambase_lr3 = perform_lr(spambase3)
    spambase_lr4 = perform_lr(spambase4)
    spambase_lr5 = perform_lr(spambase5)

    lr_perf = [spambase_lr1, spambase_lr2, spambase_lr3, spambase_lr4, spambase_lr5]

    return nb_perf, lr_perf


# Run Naive Bayes on Spambase
def nb_spambase(spambase_data, target_class):
    # Split the data set into 2/3 and 1/3
    spambase_data_train = spambase_data.sample(frac=.667)
    spambase_data_test = spambase_data.drop(spambase_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(spambase_data_train.iloc[:, 0:58], spambase_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(spambase_data_test.iloc[:, 0:58], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=spambase_data_test[target_class])

    return nb_perf


# NB and LG for Breast Cancer
def run_bc(filename, target_class, class_wanted, bc_names):
    # Setup data
    bc_obj = bc.BreastCancer()
    bc_data = bc_obj.setup_data_bc(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                   bc_names=bc_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    bc1, bc2, bc3, bc4, bc5 = five_fold.five_fold_sort_class(data=bc_data, sortby=target_class)

    bc_nb1 = nb_bc(bc_data=bc1, target_class=target_class)
    bc_nb2 = nb_bc(bc_data=bc1, target_class=target_class)
    bc_nb3 = nb_bc(bc_data=bc1, target_class=target_class)
    bc_nb4 = nb_bc(bc_data=bc1, target_class=target_class)
    bc_nb5 = nb_bc(bc_data=bc1, target_class=target_class)

    nb_perf = [bc_nb1, bc_nb2, bc_nb3, bc_nb4, bc_nb5]

    bc_lr1 = perform_lr(bc1)
    bc_lr2 = perform_lr(bc2)
    bc_lr3 = perform_lr(bc3)
    bc_lr4 = perform_lr(bc4)
    bc_lr5 = perform_lr(bc5)

    lr_perf = [bc_lr1, bc_lr2, bc_lr3, bc_lr4, bc_lr5]

    return nb_perf, lr_perf


# Run Naive Bayes on Breast Cancer
def nb_bc(bc_data, target_class):
    # Split the data set into 2/3 and 1/3
    bc_data_train = bc_data.sample(frac=.667)
    bc_data_test = bc_data.drop(bc_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(bc_data_train.iloc[:, 0:4], bc_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(bc_data_test.iloc[:, 0:4], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=bc_data_test[target_class])

    return nb_perf


# NB and LG for House Votes
def run_votes(filename, target_class, class_wanted, vote_names):
    # Setup data
    votes_obj = votes.HouseVotes()
    votes_data = votes_obj.setup_data_votes(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            vote_names=vote_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    votes1, votes2, votes3, votes4, votes5 = five_fold.five_fold_sort_class(data=votes_data, sortby=target_class)

    votes_nb1 = nb_votes(votes_data=votes1, target_class=target_class)
    votes_nb2 = nb_votes(votes_data=votes1, target_class=target_class)
    votes_nb3 = nb_votes(votes_data=votes1, target_class=target_class)
    votes_nb4 = nb_votes(votes_data=votes1, target_class=target_class)
    votes_nb5 = nb_votes(votes_data=votes1, target_class=target_class)

    nb_perf = [votes_nb1, votes_nb2, votes_nb3, votes_nb4, votes_nb5]

    votes_lr1 = perform_lr(votes1)
    votes_lr2 = perform_lr(votes2)
    votes_lr3 = perform_lr(votes3)
    votes_lr4 = perform_lr(votes4)
    votes_lr5 = perform_lr(votes5)

    lr_perf = [votes_lr1, votes_lr2, votes_lr3, votes_lr4, votes_lr5]

    return nb_perf, lr_perf


# Run Naive Bayes on House Votes
def nb_votes(votes_data, target_class):
    # Split the data set into 2/3 and 1/3
    votes_data_train = votes_data.sample(frac=.667)
    votes_data_test = votes_data.drop(votes_data_train.index)

    # Create Naive Bayes for later comparison (Added in last, so that is why it is not greatly structured in the code)
    naive = nb.NaiveBayes()
    # Train classifier
    trainer_classifier = naive.naive_bayes_train(votes_data_train.iloc[:, 0:4], votes_data_train[target_class])
    # Test classifier
    test_classifier = naive.naive_bayes_test(votes_data_test.iloc[:, 0:4], trainer_classifier)
    # Get success rate back
    nb_perf = naive.compare_prediction(predict_classier=test_classifier, data=votes_data_test[target_class])

    return nb_perf


# Run Logistic Regression
def perform_lr(data):
    # Split the data set into 2/3 and 1/3
    data_train = data.sample(frac=.667)
    data_test = data.drop(data_train.index)

    logistic = lr.LogisticRegression(learning_rate=2)
    weighted_classifier = logistic.train_lr(data=data_train)
    accuracy, prediction_list = logistic.test_lr(data=data_test)

    lr_values = [weighted_classifier, accuracy, prediction_list]

    return lr_values


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment5Output.txt", "w")

    ##### Iris #####
    iris_target_class = "class"
    class_wanted = "Iris-virginica"
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris_nb, iris_lr = run_iris(filename="data/iris.data", target_class=iris_target_class, class_wanted=class_wanted,
                                iris_names=iris_names)
    for iris_perf in iris_nb:
        print("Success rate for Iris Naive Bayes: " + str(iris_perf) + "%")
    print()
    for classifier, perf, predictions in iris_lr:
        print("Weighted Classifier for Logistic Regression:")
        print(classifier)
        print("Class predictions for Logistic Regression:")
        print(predictions)
        print("Success rate for Iris Logistic Regression: " + str(perf) + "%")
        print()
    print('\n' * 3)

    ##### Glass #####
    glass_target_class = "Type of glass"
    class_wanted = 3
    glass_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    glass_nb, glass_lr = run_glass(filename="data/glass.data", target_class=glass_target_class,
                                   class_wanted=class_wanted, glass_names=glass_names)
    for glass_perf in glass_nb:
        print("Success rate for Glass Naive Bayes: " + str(glass_perf) + "%")
    print()
    for classifier, perf, predictions in glass_lr:
        print("Weighted Classifier for Logistic Regression:")
        print(classifier)
        print("Class predictions for Logistic Regression:")
        print(predictions)
        print("Success rate for Glass Logistic Regression: " + str(perf) + "%")
        print()
    print('\n' * 3)

    ##### Spambase #####
    spambase_target_class = "57"
    spambase_nb, spambase_lr = run_spambase(filename="data/spambase.data", target_class=spambase_target_class)
    for spambase_perf in spambase_nb:
        print("Success rate for Spambase Naive Bayes: " + str(spambase_perf) + "%")
    print()
    for classifier, perf, predictions in spambase_lr:
        print("Weighted Classifier for Logistic Regression:")
        print(classifier)
        print("Class predictions for Logistic Regression:")
        print(predictions)
        print("Success rate for Spambase Logistic Regression: " + str(perf) + "%")
        print()
    print('\n' * 3)

    ##### Breast Cancer #####
    bc_target_class = "Class"
    bc_class_wanted = 4
    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    bc_nb, bc_lr = run_bc(filename="data/breast-cancer-wisconsin.data", target_class=bc_target_class,
                          class_wanted=bc_class_wanted, bc_names=breast_cancer_names)

    for nb_perf in bc_nb:
        print("Success rate for Breast Cancer Naive Bayes: " + str(nb_perf) + "%")
    print()
    for classifier, perf, predictions in bc_lr:
        print("Weighted Classifier for Logistic Regression:")
        print(classifier)
        print("Class predictions for Logistic Regression:")
        print(predictions)
        print("Success rate for Breast Cancer Logistic Regression: " + str(perf) + "%")
        print()
    print('\n' * 3)

    ##### House Votes #####
    votes_target_class = "class"
    votes_class_wanted = "republican"
    votes_names = ["class", "handicapped", "water", "adoption", "physician", "el-salvador", "religious",
                   "anti", "aid", "mx", "immigration", "synfuels", "education", "superfund", "crime",
                   "duty-free", "export"]
    votes_nb, votes_lr = run_votes(filename="data/house-votes-84.data", target_class=votes_target_class,
                                   class_wanted=votes_class_wanted, vote_names=votes_names)
    for nb_perf in votes_nb:
        print("Success rate for House Votes Naive Bayes: " + str(nb_perf) + "%")
    print()
    for classifier, perf, predictions in votes_lr:
        print("Weighted Classifier for Logistic Regression:")
        print(classifier)
        print("Class predictions for Logistic Regression:")
        print(predictions)
        print("Success rate for House Votes Logistic Regression: " + str(perf) + "%")
        print()
    print('\n' * 3)


if __name__ == "__main__":
    main()
