import numpy as np

"""
    This class performs everything related to Logistic Regression.
"""


class LogisticRegression:
    def __init__(self, learning_rate):
        self.class_weights = []
        self.learning_rate = learning_rate

    # Train the model for logistic regression (pass in one hot encoded data)
    def train_lr(self, data):
        # Put data frame into data array
        data_array = data.values

        # Setup class weights for the two possible classes(1. it is not the desired class, 2. it is the desired class)
        self.class_weights = np.random.randint(low=-1, high=2, size=(2, data.shape[1] - 1)) / 100

        # Used to get us out of the loop
        weights_have_changed = True

        # Loop until the weights stay the same
        while weights_have_changed:
            comparable_class_weights = np.zeros((2, data.shape[1] - 1))

            # Iterate through data points in data array
            for point_index in range(0, len(data_array)):
                # Get data point from data array
                data_point = data_array[point_index]
                # Get data point without class value
                data_point_shortened = data_point[:-1]
                # Get data point class value
                data_point_class = data_point[-1]
                # Get the cumulative class values
                cumulative_class = self.get_weighted_sum(data_point_shortened=data_point_shortened)
                # Get the sum of the cumulative class values
                sum_of_class_sums = cumulative_class.sum()
                # Get the class prediction values based on cumulative class values and weight sum
                predictions = self.get_class_predictions(sum_of_class_sums=sum_of_class_sums,
                                                         cumulative_class=cumulative_class)
                # Setup matching weights for the two possible classes
                matching_weight = np.zeros(2)
                matching_weight[int(data_point_class)] = 1

                # Get the prediction error, and realign how it is shaped
                prediction_error = matching_weight - predictions
                prediction_error = np.reshape(prediction_error, (len(prediction_error), -1))

                # Update prediction error by multiplying by all features in the data point
                prediction_value = prediction_error * data_point_shortened

                # Add the latest class weight and prediction value together to be used after iterating through all
                # data points
                comparable_class_weights = np.add(comparable_class_weights, prediction_value)

            # Update the class weights with the latest and the learning rate
            self.class_weights = self.class_weights + (self.learning_rate * comparable_class_weights)

            # If the comparable class weights have any values other than 0, we are finished
            if comparable_class_weights.any():
                weights_have_changed = False

        return self.class_weights

    # Compute weighted sum of data point and class weights
    def get_weighted_sum(self, data_point_shortened):
        return np.dot(self.class_weights, data_point_shortened)

    # Get class predictions
    def get_class_predictions(self, sum_of_class_sums, cumulative_class):
        return np.apply_along_axis(func1d=lambda x: (x / sum_of_class_sums if sum_of_class_sums else [0.0, 0.0]),
                                   axis=0, arr=cumulative_class)

    # Test the predictions data with a model that is already trained
    def test_lr(self, data):
        prediction_list = []
        actual_class_list = []
        # Put data frame into data array
        data_array = data.values

        for point_index in range(0, len(data_array)):
            # Get data point from data array
            data_point = data_array[point_index]
            # Get data point without class value
            data_point_shortened = data_point[:-1]
            # Get data point class value
            data_point_class = data_point[-1]

            # Append actual class value to list
            actual_class_list.append(data_point_class)

            # Make prediction
            prediction = self.predict_classes(data_point_shortened)
            # Append class prediction to list
            prediction_list.append(prediction)

        # Compute accuracy
        correct = 0
        for index in range(0, len(actual_class_list)):
            if actual_class_list[index] == prediction_list[index]:
                correct += 1

        lr_accuracy = correct / len(actual_class_list)
        lr_accuracy = lr_accuracy * 100

        return lr_accuracy, prediction_list

    # Run prediction
    def predict_classes(self, data_point):
        value = np.apply_along_axis(lambda x: np.exp(np.dot(x, data_point) + 1), 1, self.class_weights)
        max_value = value.argmax()
        return max_value
