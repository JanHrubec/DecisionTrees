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

        if args.dataset == 'wine':
            values, counts = np.unique(self.target, return_counts=True)
            self.value = values[np.argmax(counts)]
        else:
            self.value = np.mean(self.target)
        
        # Find best split
        n_samples, n_features = self.data.shape
        for feature in range(n_features):
            sorted_indices = np.argsort(self.data[:, feature])
            sorted_data = self.data[sorted_indices]
            sorted_target = self.target[sorted_indices]

            for i in range(1, n_samples):
                if sorted_data[i, feature] == sorted_data[i-1, feature]:
                    continue

                threshold = (sorted_data[i, feature] + sorted_data[i - 1, feature]) / 2
                left_mask = self.data[:, feature] < threshold
                right_mask = ~left_mask
                left_target, right_target = self.target[left_mask], self.target[right_mask]

                if len(left_target) == 0 or len(right_target) == 0:
                    continue

                if args.dataset == 'wine':
                    left_c = entropy(left_target)
                    right_c = entropy(right_target)
                    c = entropy(self.target)
                else:
                    left_c = squared_error(np.mean(left_target), left_target)
                    right_c = squared_error(np.mean(right_target), right_target)
                    c = squared_error(np.mean(self.target), self.target)

                delta_c = c - (left_c * len(left_target) + right_c * len(right_target)) / n_samples
                if delta_c > self.delta_c:
                    self.delta_c = delta_c
                    self.feature = feature
                    self.threshold = threshold
                    self.left_data = self.data[left_mask]
                    self.right_data = self.data[right_mask]
                    self.left_target = left_target
                    self.right_target = right_target

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
    root = Node(train_data, train_target, 0)
    if args.max_leaves is None:
        recursive_split(root)
    else:
        pq = PriorityQueue()
        pq.put((-root.delta_c, root))
        leaves = 1
        while not pq.empty():
            delta_c, node = pq.get()
            if leaves >= args.max_leaves or (args.max_depth is not None and node.depth >= args.max_depth) or (args.min_to_split is not None and len(node.data) < args.min_to_split) or node.delta_c == 0:
                break
            left, right = node.split()
            pq.put((-left.delta_c, left))
            pq.put((-right.delta_c, right))
            leaves += 1
    return root

def predict(node, data):
    if node.left is None or node.right is None:
        return node.value
    if data[node.feature] < node.threshold:
        return predict(node.left, data)
    else:
        return predict(node.right, data)

def main(args: argparse.Namespace):
    # Načtení datasetu.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # TODO: Rozdělte dataset na trénovací a testovací část, funkci z knihovny sklearn
    # předejte argumenty `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Implementace binárního rozhodovacího stromu
    #
    # - Budete implementovat strom pro klasifikaci i regresi podle typu
    #   datasetu. `wine` dataset je pro klasifikaci a `diabetes` pro regresi.
    #
    # - Pro klasifikaci: Pro každý list predikujte třídu, která je nejčastější
    #   (a pokud je těchto tříd několik, vyberte tu s nejmenším číslem).
    # - Pro regresi: Pro každý list predikujte průměr target (cílových) hodnot.
    #    
    # - Pro klasifikaci použijte jako kritérium entropy kritérium a pro regresi
    #   použijte jako kritérium SE (squared error).
    #
    # - Pro rozdělení vrcholu vyzkoušejte postupně všechny featury. Pro každou
    #   featuru vyzkoušejte všechna možná místa na rozdělení seřazená vzestupně
    #   a rozdělte vrchol na místě, který nejvíce sníží kritérium.
    #   Pokud existuje takových míst několik, vyberte první z nich.
    #   Každé možné místo na rozdělení je průměrem dvou nejbližších unikátních hodnot
    #   featur z dat odpovídající danému vrcholu.
    #   Např. pro čtyři data s hodnotami featur 1, 7, 3, 3 jsou místa na rozdělení 2 a 5.
    #
    # - Rozdělení vrcholu povolte pouze pokud:
    #   - pokud hloubka vrcholu je menší než `args.max_depth`
    #     (hloubka kořenu je nula). Pokud `args.max_depth` je `None`,
    #     neomezujte hloubku stromu.
    #   - pokud je méně než `args.max_leaves` listů ve stromě
    #     (list je vrchol stromu bez synů). Pokud `args.max_leaves` je `None`,
    #     neomezujte počet listů.
    #   - je alespoň `args.min_to_split` dat v daném vrcholu.
    #   - hodnota kritéria není nulová.
    #
    # - Pokud `args.max_leaves` není `None`: opakovaně rozdělujte listové vrcholy,
    #   které splňují podmínky na rozdělení vrcholu a celková hodnota kritéria
    #   ($c_{levý syn} + c_{pravý syn} - c_{rodič}$) nejvíce klesne.
    #   Pokud je několik takových vrcholů, vyberte ten, který byl vytvořen dříve
    #   (levý syn je považován, že je vytvořený dříve než pravý syn).
    #
    #   Pokud `args.max_leaves` je `None`, použijte rekurzivní přístup (nejdříve rozdělujte
    #   levé syny, poté pravé syny) a rozdělte každý vrchol, který splňuje podmínky.
    #   Tento rekurzivní přístup není důležitý v této úloze, ale až v nasledující
    #   úloze - Random Forest.
    #
    # - Nakonec vypočítejte trénovací a testovací chybu
    #   - RMSE, root mean squared error, pro regresi
    #   - accuracy pro klasifikaci
    
    root = build_tree(train_data, train_target)
    
    train_predictions = [predict(root, x) for x in train_data]
    test_predictions = [predict(root, x) for x in test_data]

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
