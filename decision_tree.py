#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from queue import PriorityQueue

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="wine", type=str, help="Použitý dataset; buď `wine` nebo `diabetes`",
                    choices=["wine", "diabetes"])
parser.add_argument("--max_depth", default=None, type=int, help="Maximální hloubka rozhodovacího stromu")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximální počet listů stromu")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimální počet dat pro rozdělení vrcholu (listu)")
parser.add_argument("--seed", default=42, type=int, help="Náhodný seed")
parser.add_argument("--test_size", default=0.25, type=float, help="Velikost testovací množiny")

def entropy(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target)
    return - len(target) * np.sum(probabilities * np.log2(probabilities))

def squared_error(pred, target):
    return np.sum((target - pred) ** 2)

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
        if args.dataset == 'wine':
            values, counts = np.unique(self.target, return_counts=True)
            self.value = values[np.argmax(counts)]
        else:
            self.value = np.mean(self.target)
        
        # Find best split
        n_samples, n_features = self.data.shape
        #loop through all features
        for feature in range(n_features):
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
                if args.dataset == 'wine':
                    left_c = entropy(left_target)
                    right_c = entropy(right_target)
                    c = entropy(self.target)
                else:
                    left_c = squared_error(np.mean(left_target), left_target)
                    right_c = squared_error(np.mean(right_target), right_target)
                    c = squared_error(np.mean(self.target), self.target)

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

    #split node into two child nodes
    def split(self):
        left_node = Node(self.left_data, self.left_target, self.depth + 1)
        right_node = Node(self.right_data, self.right_target, self.depth + 1)
        self.left = left_node
        self.right = right_node
        return left_node, right_node

def recursive_split(current):
    if (args.max_depth is not None and current.depth >= args.max_depth) or (args.min_to_split is not None and len(current.data) < args.min_to_split) or current.delta_c == 0:
        return
    left, right = current.split()
    if left and right:
        recursive_split(left)
        recursive_split(right)

def build_tree(train_data, train_target):
    #create root node
    root = Node(train_data, train_target, 0)
    if args.max_leaves is None:
        #split nodes recusively
        recursive_split(root)
    else:
        #split nodes using priority queue based on largest delta c
        counter = 0
        pq = PriorityQueue()
        #add root to queue
        pq.put((-root.delta_c, counter, root))
        leaves = 1
        while not pq.empty() and leaves < args.max_leaves:
            #get node with highest delta c
            delta_c, _, node = pq.get()
            if (args.max_depth is not None and node.depth >= args.max_depth) or (args.min_to_split is not None and len(node.data) < args.min_to_split) or node.delta_c == 0:
                continue
            #if node can be split, split it
            left, right = node.split()
            #add children to queue
            pq.put((-left.delta_c, counter, left))
            pq.put((-right.delta_c, counter+1, right))
            counter += 2
            leaves += 1
    return root

def predict(node, data):
    # if node is leaf, return its value
    if node.left is None:
        return node.value
    # if feature value is less than threshold, go left, otherwise go right
    if data[node.feature] < node.threshold:
        return predict(node.left, data)
    else:
        return predict(node.right, data)

def main(args: argparse.Namespace):
    # load dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    #split data
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)
    
    #build tree
    root = build_tree(train_data, train_target)
    
    #get predictions
    train_predictions = [predict(root, x) for x in train_data]
    test_predictions = [predict(root, x) for x in test_data]

    #calculate error
    if args.dataset == 'diabetes':
        train_rmse = mean_squared_error(train_target, train_predictions, squared=False)
        test_rmse = mean_squared_error(test_target, test_predictions, squared = False)
        print("Train RMSE: {:.5f}".format(train_rmse))
        print("Test RMSE: {:.5f}".format(test_rmse))
    else:
        train_accuracy = accuracy_score(train_target, train_predictions)
        test_accuracy = accuracy_score(test_target, test_predictions)
        print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
        print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
