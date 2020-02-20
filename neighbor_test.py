import math
import statistics as stat
import pandas as pd
from nearest_neighbor import Neighborhood
from calculations import mean_square_error


def user_read(path):
    user_frame = pd.DataFrame()
    with open(path, "r") as file:
        for line in file:
            user, movie, rating, time = line.split()
            if user not in user_frame.index:
                row = pd.Series(name=user, dtype="float64")
                user_frame = user_frame.append(row)
            if movie not in user_frame.columns:
                user_frame.insert(0, movie, float("nan"))
            user_frame.loc[user][movie] = rating
    user_frame = user_frame.reindex([str(i) for i in range(max(int(c) for c in user_frame.columns))], axis=1)
    return user_frame


if __name__ == "__main__":

    # Read in train and test data
    train_data = user_read("data/u1-base.base")
    test_data = user_read("data/u1-test.test")

    # Make sure training and test data have the same features
    test_data = test_data.reindex(train_data.columns, axis=1)

    # Fill NaN in training with "average" rating (2)
    train_data = train_data.fillna(2)

    # K = 3 K-nearest neighbors
    neighborhood = Neighborhood(3)
    neighborhood.fit(train_data)
    results = pd.DataFrame(index=train_data.index, columns=train_data.columns)
    mse_values = []

    # predict values for test data
    for index, example in train_data.iterrows():
        test_features = list(filter(lambda x: not math.isnan(example[x]), train_data.columns))
        for feature in test_features:
            results[index][feature] = neighborhood.predict(example, feature, default=2)
        mse_values.append(mean_square_error(example, results[index], train_data.columns))
    print(stat.mean(mse_values))
