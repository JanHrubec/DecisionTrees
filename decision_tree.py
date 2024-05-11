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
    #calculate entropy criterion
    _sum = 0
    for i in set(target):
        p = sum(target == i) / len(target)
        _sum += p * np.log2(p)
    
    return -len(target) * _sum

def squared_error(pred, target):
    #calculate squared error criterion
    _sum = 0
    for i in range(len(target)):
        _sum += (target[i] - pred) ** 2
    return _sum

class Node:
    def __init__(self, data, target, depth):
        self.data = data
        self.target = target
        self.depth = depth
        self.delta_c = float('-inf')
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

        if args.dataset == 'wine':
            self.value = max(set(target), key=target.count)
        else:
            self.value = np.mean(target)
        
        #find best split
        #loop through features
        for feature in range(len(self.data[0])):
            thresholds = []
            #sort based on feature
            sorted_data = sorted(self.data, key=lambda x: x[feature])
            #loop through sorted data and find thresholds as average of two unique consecutive values
            for i in range(len(sorted_data)-1):
                if sorted_data[i][feature] != sorted_data[i+1][feature]:
                    thresholds.append((sorted_data[i][feature] + sorted_data[i+1][feature]) / 2)
            
            #loop through thresholds
            for t in thresholds:
                #split into two sons
                left_data = []
                right_data = []
                left_target = []
                right_target = []
                for i in range(len(self.data)):
                    if self.data[i][feature] < t:
                        left_data.append(self.data[i])
                        left_target.append(self.target[i])
                    else:
                        right_data.append(self.data[i])
                        right_target.append(self.target[i])

                #calculate delta_c
                if args.dataset == 'wine':
                    left_c = entropy(left_target)
                    right_c = entropy(right_target)
                    c = entropy(self.target)
                else:
                    left_c = squared_error(np.mean(left_target), left_target)
                    right_c = squared_error(np.mean(right_target), right_target)
                    c = squared_error(self.value, self.target)

                delta_c = left_c + right_c - c
                #if delta_c is bigger than current best, update best split
                if delta_c > self.delta_c:
                    self.delta_c = delta_c
                    self.feature = feature
                    self.threshold = t

    def split(self):
        #split data
        left_data = []
        right_data = []
        left_target = []
        right_target = []
        for i in range(len(self.data)):
            if self.data[i][self.feature] < self.threshold:
                left_data.append(self.data[i])
                left_target.append(self.target[i])
            else:
                right_data.append(self.data[i])
                right_target.append(self.target[i])

        #create sons
        self.left = Node(left_data, left_target, self.depth + 1)
        self.right = Node(right_data, right_target, self.depth + 1)

        return self.left, self.right

def recursive_split(current):
    if((args.max_depth is not None and current.depth >= args.max_depth) or (args.min_to_split is not None and len(current.data) < args.min_to_split)):
        return
    
    left, right = current.split()

    recursive_split(left)
    recursive_split(right)

def main(args: argparse.Namespace):
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)
    
    #training
    root = Node(train_data, train_target, 0)

    if(args.max_leaves is None):
        recursive_split(root)
    else:
        pq = PriorityQueue()
        #push root to priority queue
        pq.put((-root.delta_c, root))
        leaves_left = args.max_leaves
        while(leaves_left > 0):
            current = pq.get()[1]
            left, right = current.split()
            pq.put((-left.delta_c, left))
            pq.put((-right.delta_c, right))
            leaves_left -= 1 #remove one leaf (current) and add two new leaves
        
    #predicting
    pred_train = []
    #for each data, traverse the tree
    for i in range(len(train_data)):
        current = root
        #while it has children, go to the correct one
        while(current.left is not None):
            if(train_data[i][current.feature] < current.threshold):
                current = current.left
            else:
                current = current.right
        #add the prediction to the list
        pred_train.append(current.value)

    pred_test = []
    #for each data, traverse the tree
    for i in range(len(test_data)):
        current = root
        #while it has children, go to the correct one
        while(current.left is not None):
            if(train_data[i][current.feature] < current.threshold):
                current = current.left
            else:
                current = current.right
        #add the prediction to the list
        pred_test.append(current.value)

    if args.dataset == 'diabetes':
        train_rmse = mean_squared_error(train_target, pred_train)
        test_rmse = mean_squared_error(test_target, pred_test)
        print("Train RMSE: {:.5f}".format(train_rmse))
        print("Test RMSE: {:.5f}".format(test_rmse))
    else:
        train_accuracy = accuracy_score(train_target, pred_train)
        test_accuracy = accuracy_score(test_target, pred_test)
        print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
        print("Test accuracy: {:.1f}%".format(100 * test_accuracy))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
