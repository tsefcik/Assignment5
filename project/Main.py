from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Spambase as spambase
from project import BreastCancer as bc
from project import HouseVotes as votes
from project import NaiveBayes as nb
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

    return iris_data, nb_perf


# NB and LG for Glass
def run_glass(filename, target_class, class_wanted, glass_names):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            glass_names=glass_names)

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

    return glass_data, nb_perf


# NB and LG for Spambase
def run_spambase(filename, target_class):
    # Setup data
    spambase_obj = spambase.Spambase()
    spambase_data = spambase_obj.setup_data_spambase(filename=filename, target_class=target_class)

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

    return spambase_data, nb_perf


# NB and LG for Breast Cancer
def run_bc(filename, target_class, class_wanted, bc_names):
    # Setup data
    bc_obj = bc.BreastCancer()
    bc_data = bc_obj.setup_data_bc(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                   bc_names=bc_names)
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

    return bc_data, nb_perf


# NB and LG for House Votes
def run_votes(filename, target_class, class_wanted, vote_names):
    # Setup data
    votes_obj = votes.HouseVotes()
    votes_data = votes_obj.setup_data_votes(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            vote_names=vote_names)
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

    return votes_data, nb_perf


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment5Output.txt", "w")

    ##### Iris #####
    iris_target_class = "class"
    class_wanted = "Iris-virginica"
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris_data, iris_nb = run_iris(filename="data/iris.data", target_class=iris_target_class, class_wanted=class_wanted,
                                  iris_names=iris_names)
    print("Success rate for Iris Naive Bayes: " + str(iris_nb) + "%")
    print('\n' * 3)

    ##### Glass #####
    glass_target_class = "Type of glass"
    class_wanted = 3
    glass_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    glass_data, glass_nb = run_glass(filename="data/glass.data", target_class=glass_target_class,
                                     class_wanted=class_wanted, glass_names=glass_names)
    print("Success rate for Glass Naive Bayes: " + str(glass_nb) + "%")
    print('\n' * 3)

    ##### Spambase #####
    spambase_target_class = "57"
    spambase_data, spambase_nb = run_spambase(filename="data/spambase.data", target_class=spambase_target_class)
    print("Success rate for Spambase Naive Bayes: " + str(spambase_nb) + "%")
    print('\n' * 3)

    ##### Breast Cancer #####
    bc_target_class = "Class"
    bc_class_wanted = 4
    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    bc_data, bc_nb = run_bc(filename="data/breast-cancer-wisconsin.data", target_class=bc_target_class,
                            class_wanted=bc_class_wanted, bc_names=breast_cancer_names)
    print("Success rate for Breast Cancer Naive Bayes: " + str(bc_nb) + "%")
    print('\n' * 3)

    ##### House Votes #####
    votes_target_class = "class"
    votes_class_wanted = "republican"
    votes_names = ["class", "handicapped", "water", "adoption", "physician", "el-salvador", "religious",
                   "anti", "aid", "mx", "immigration", "synfuels", "education", "superfund", "crime",
                   "duty-free", "export"]
    votes_data, votes_nb = run_votes(filename="data/house-votes-84.data", target_class=votes_target_class,
                                     class_wanted=votes_class_wanted, vote_names=votes_names)
    print("Success rate for House Votes Naive Bayes: " + str(votes_nb) + "%")
    print('\n' * 3)


if __name__ == "__main__":
    main()
