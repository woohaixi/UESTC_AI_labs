import math
import pandas as pd
from collections import Counter


# 计算数据集的熵（保持不变）
def entropy(data, label_col):
    labels = data[label_col]
    total = len(labels)
    if total == 0:
        return 0
    label_counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in label_counts.values() if count > 0)


def calculate_gain_ratio(data, attr, label_col):
    total_entropy = entropy(data, label_col)
    if attr in data.columns and pd.api.types.is_numeric_dtype(data[attr]):
        values = sorted(data[attr])
        split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        best_gain_ratio = -float('inf')
        best_split = None
        for split in split_points:
            left = data[data[attr] <= split]
            right = data[data[attr] > split]
            info_a = (len(left) / len(data)) * entropy(left, label_col) + (len(right) / len(data)) * entropy(right,
                                                                                                             label_col)
            gain = total_entropy - info_a
            left_ratio = len(left) / len(data)
            right_ratio = len(right) / len(data)
            iv = 0
            if left_ratio > 0:
                iv += -left_ratio * math.log2(left_ratio)
            if right_ratio > 0:
                iv += -right_ratio * math.log2(right_ratio)
            if iv == 0:
                gain_ratio_val = gain
            else:
                gain_ratio_val = gain / iv
            if gain_ratio_val > best_gain_ratio:
                best_gain_ratio = gain_ratio_val
                best_split = split
        return best_gain_ratio, best_split
    else:
        values = data[attr].unique()
        info_a = 0
        iv = 0
        for val in values:
            subset = data[data[attr] == val]
            prob = len(subset) / len(data)
            info_a += prob * entropy(subset, label_col)
            if prob > 0:
                iv += -prob * math.log2(prob)
        gain = total_entropy - info_a
        if iv == 0:
            return gain, None
        return gain / iv, None

class C45Node:
    def __init__(self, attribute=None, threshold=None, label=None, branches=None, depth=0):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.branches = branches if branches is not None else {}
        self.depth = depth


def build_c45_tree(data, attributes, label_col, max_depth=None, min_samples_split=2, current_depth=0):
    labels = data[label_col]
    if len(labels.unique()) == 1:
        return C45Node(label=labels.iloc[0], depth=current_depth)

    if len(data) < min_samples_split or (max_depth is not None and current_depth >= max_depth):
        return C45Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth)

    if not attributes:
        return C45Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth)

    gains = [(attr, calculate_gain_ratio(data, attr, label_col)[0]) for attr in attributes]
    best_attr, _ = max(gains, key=lambda x: x[1])

    if pd.api.types.is_numeric_dtype(data[best_attr]):
        _, best_split = calculate_gain_ratio(data, best_attr, label_col)
        node = C45Node(attribute=best_attr, threshold=best_split, depth=current_depth)
        left_data = data[data[best_attr] <= best_split]
        right_data = data[data[best_attr] > best_split]
        if len(left_data) > 0:
            node.branches['<='] = build_c45_tree(left_data, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                      current_depth + 1)
        if len(right_data) > 0:
            node.branches['>'] = build_c45_tree(right_data, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                     current_depth + 1)
    else:
        node = C45Node(attribute=best_attr, depth=current_depth)
        for val in data[best_attr].unique():
            subset = data[data[best_attr] == val]
            if len(subset) == 0:
                node.branches[val] = C45Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth + 1)
            else:
                node.branches[val] = build_c45_tree(subset, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                         current_depth + 1)
    return node

def predict_with_c45(tree, instance):
    if tree.label is not None:
        return tree.label
    attr = tree.attribute
    if tree.threshold is not None:
        if instance[attr] <= tree.threshold:
            return predict_with_c45(tree.branches['<='], instance)
        else:
            return predict_with_c45(tree.branches['>'], instance)
    else:
        val = instance[attr]
        if val not in tree.branches:
            return max(tree.branches.values(),
                       key=lambda x: Counter([predict_with_c45(x, instance) for _ in range(1)]).most_common(1)[0][1]).label
        return predict_with_c45(tree.branches[val], instance)

def main():
    train_data = pd.read_csv('traindata.txt', sep='\s+', header=None,
                             names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])
    test_data = pd.read_csv('testdata.txt', sep='\s+', header=None, names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])

    attributes = set(train_data.columns) - {'label'}
    label_col = 'label'

    # 更新函数调用名
    tree = build_c45_tree(train_data, attributes, label_col, max_depth=5, min_samples_split=2)

    correct = 0
    total = len(test_data)
    for _, instance in test_data.iterrows():
        prediction = predict_with_c45(tree, instance)
        if prediction == instance[label_col]:
            correct += 1

    accuracy = correct / total * 100
    print(f"分类准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()