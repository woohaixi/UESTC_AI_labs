import math
import pandas as pd
from collections import Counter


# 计算数据集的基尼指数
def gini_index(data, label_col):
    labels = data[label_col]
    total = len(labels)
    if total == 0:
        return 0
    label_counts = Counter(labels)
    return 1 - sum((count / total) ** 2 for count in label_counts.values())


# 计算某个属性的基尼指数增益
def gini_gain(data, attr, label_col):
    total_gini = gini_index(data, label_col)
    if attr in data.columns and pd.api.types.is_numeric_dtype(data[attr]):
        # 处理连续属性
        values = sorted(data[attr])
        split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        best_gini_gain = -float('inf')
        best_split = None
        for split in split_points:
            left = data[data[attr] <= split]
            right = data[data[attr] > split]
            gini_a = (len(left) / len(data)) * gini_index(left, label_col) + \
                     (len(right) / len(data)) * gini_index(right, label_col)
            gini_gain_val = total_gini - gini_a
            if gini_gain_val > best_gini_gain:
                best_gini_gain = gini_gain_val
                best_split = split
        return best_gini_gain, best_split
    else:
        # 处理离散属性
        values = data[attr].unique()
        gini_a = 0
        for val in values:
            subset = data[data[attr] == val]
            prob = len(subset) / len(data)
            gini_a += prob * gini_index(subset, label_col)
        gini_gain_val = total_gini - gini_a
        return gini_gain_val, None


# 表示决策树节点的类
class Node:
    def __init__(self, attribute=None, threshold=None, label=None, branches=None, depth=0):
        self.attribute = attribute  # 分裂属性
        self.threshold = threshold  # 连续属性的阈值
        self.label = label         # 叶节点标签
        self.branches = branches if branches is not None else {}  # 子节点
        self.depth = depth         # 当前节点深度


# CART算法构建决策树，使用基尼指数
def cart(data, attributes, label_col, max_depth=None, min_samples_split=2, current_depth=0):
    labels = data[label_col]
    # 如果所有样本的标签相同，返回叶节点
    if len(labels.unique()) == 1:
        return Node(label=labels.iloc[0], depth=current_depth)

    # 如果样本数少于最小分裂阈值，或达到最大深度，返回多数类标签的叶节点
    if len(data) < min_samples_split or (max_depth is not None and current_depth >= max_depth):
        return Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth)

    # 如果没有可用属性，返回多数类标签的叶节点
    if not attributes:
        return Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth)

    # 找到最佳分裂属性
    gains = [(attr, gini_gain(data, attr, label_col)[0]) for attr in attributes]
    best_attr, _ = max(gains, key=lambda x: x[1])

    # 为最佳属性创建节点
    if pd.api.types.is_numeric_dtype(data[best_attr]):
        # 处理连续属性
        _, best_split = gini_gain(data, best_attr, label_col)
        node = Node(attribute=best_attr, threshold=best_split, depth=current_depth)
        left_data = data[data[best_attr] <= best_split]
        right_data = data[data[best_attr] > best_split]
        if len(left_data) > 0:
            node.branches['<='] = cart(left_data, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                      current_depth + 1)
        if len(right_data) > 0:
            node.branches['>'] = cart(right_data, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                     current_depth + 1)
    else:
        # 处理离散属性
        node = Node(attribute=best_attr, depth=current_depth)
        for val in data[best_attr].unique():
            subset = data[data[best_attr] == val]
            if len(subset) == 0:
                node.branches[val] = Node(label=Counter(labels).most_common(1)[0][0], depth=current_depth + 1)
            else:
                node.branches[val] = cart(subset, attributes - {best_attr}, label_col, max_depth, min_samples_split,
                                         current_depth + 1)

    return node


# 使用决策树对单个实例进行分类
def classify(tree, instance):
    if tree.label is not None:
        return tree.label
    attr = tree.attribute
    if tree.threshold is not None:
        # 连续属性
        if instance[attr] <= tree.threshold:
            return classify(tree.branches['<='], instance)
        else:
            return classify(tree.branches['>'], instance)
    else:
        # 离散属性
        val = instance[attr]
        if val not in tree.branches:
            # 如果训练中未见的值，返回此节点多数标签
            return max(tree.branches.values(),
                       key=lambda x: Counter([classify(x, instance) for _ in range(1)]).most_common(1)[0][1]).label
        return classify(tree.branches[val], instance)


# 主函数运行实验
def main():
    # 加载训练和测试数据
    # 假设数据文件为用空格分隔的格式，包含5列（4个特征+1个标签）
    train_data = pd.read_csv('traindata.txt', sep='\s+', header=None,
                             names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])
    test_data = pd.read_csv('testdata.txt', sep='\s+', header=None, names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])

    # 定义属性和标签列
    attributes = set(train_data.columns) - {'label'}
    label_col = 'label'

    # 构建决策树，设置最大深度为5，最小分裂样本数为2
    tree = cart(train_data, attributes, label_col, max_depth=5, min_samples_split=2)

    # 分类测试数据并计算准确率
    correct = 0
    total = len(test_data)
    for _, instance in test_data.iterrows():
        prediction = classify(tree, instance)
        if prediction == instance[label_col]:
            correct += 1

    accuracy = correct / total * 100
    print(f"分类准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()