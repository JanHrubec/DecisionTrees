#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--bagging", default=False, action="store_true", help="Použít bagging")
parser.add_argument("--dataset", default="digits", type=str, help="Použitý dataset")
parser.add_argument("--feature_subsampling", default=1, type=float, help="Pravděpodobnost vybrání jednolivých featur")
parser.add_argument("--max_depth", default=None, type=int, help="Maximální hloubka rozhodovacího stromu")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")
parser.add_argument("--test_size", default=0.25, type=float, help="Velikost testovací množiny")
parser.add_argument("--trees", default=1, type=int, help="Počet stromů použitých v random forestu")

def main(args: argparse.Namespace):
    #load dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    #split dataset
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    #returns a random mask for features
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    #returns a random bootstrap dataset
    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    def entropy(target):
        _, counts = np.unique(target, return_counts=True)
        probabilities = counts / len(target)
        return - len(target) * np.sum(probabilities * np.log2(probabilities))

    class Node:
        def __init__(self, data, target, depth):
            self.data = np.array(data)
            self.target = np.array(target)
            self.depth = depth
            self.delta_c = float('-inf')
            self.feature = None
            self.threshold = None
            self.left_data = None
            self.right_data = None
            self.left_target = None
            self.right_target = None
            self.left = None
            self.right = None

            # Find what value this node should predict
            values, counts = np.unique(self.target, return_counts=True)
            self.value = values[np.argmax(counts)]

        #split node into two child nodes
        def split(self):
            # Find best split
            n_samples, n_features = self.data.shape

            #generate mask for features
            subsampling = subsample_features(n_features)

            #loop through all features
            for feature in range(n_features):
                #skip feature if not subsampled
                if not subsampling[feature]:
                    continue

                #sort data by value of feature
                sorted_indices = np.argsort(self.data[:, feature])
                sorted_data = self.data[sorted_indices]
                sorted_target = self.target[sorted_indices]

                #loop through all samples pairs of sorted data
                for i in range(0, n_samples-1):
                    #consider only consecutive unique values
                    if sorted_data[i, feature] == sorted_data[i + 1, feature]:
                        continue
                    
                    #calculate threshold as the average of two distinct consecutive values
                    threshold = (sorted_data[i, feature] + sorted_data[i + 1, feature]) / 2
                    #create masks for left and right data
                    left_mask = self.data[:, feature] < threshold
                    right_mask = ~left_mask
                    left_target, right_target = self.target[left_mask], self.target[right_mask]

                    if len(left_target) == 0 or len(right_target) == 0:
                        continue
                    
                    #calculate criterion for both sides
                    left_c = entropy(left_target)
                    right_c = entropy(right_target)
                    c = entropy(self.target)

                    #calculate delta c
                    delta_c = c - (left_c + right_c)
                    #if best delta c, update split
                    if delta_c > self.delta_c:
                        self.delta_c = delta_c
                        self.feature = feature
                        self.threshold = threshold
                        self.left_data = self.data[left_mask]
                        self.right_data = self.data[right_mask]
                        self.left_target = left_target
                        self.right_target = right_target

            #split node based on best delta c
            left_node = Node(self.left_data, self.left_target, self.depth + 1)
            right_node = Node(self.right_data, self.right_target, self.depth + 1)
            self.left = left_node
            self.right = right_node
            return left_node, right_node

    def recursive_split(current):
        if (args.max_depth is not None and current.depth >= args.max_depth) or current.delta_c == 0:
            return
        left, right = current.split()
        if left and right:
            recursive_split(left)
            recursive_split(right)


    def predict(node, data):
        # if node is leaf, return its value
        if node.left is None:
            return node.value
        # if feature value is less than threshold, go left, otherwise go right
        if data[node.feature] < node.threshold:
            return predict(node.left, data)
        else:
            return predict(node.right, data)

    def predict_voting(roots, data):
        #loop through all trees and get predictions
        predictions = [predict(r, data) for r in roots]
        #get most common prediction
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]

    #Create root nodes
    roots = []
    for i in range(args.trees):
        if args.bagging:
            #if bagging, create bootstrap dataset
            dataset_indices = bootstrap_dataset(train_data)
            r = Node(train_data[dataset_indices], train_target[dataset_indices], 0)
            recursive_split(r)
            roots.append(r)
        else:
            r = Node(train_data, train_target, 0)
            recursive_split(r)
            roots.append(r)

    #get predictions
    train_predictions = [predict_voting(roots, x) for x in train_data]
    test_predictions = [predict_voting(roots, x) for x in test_data]

    train_accuracy = accuracy_score(train_target, train_predictions)
    test_accuracy = accuracy_score(test_target, test_predictions)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
