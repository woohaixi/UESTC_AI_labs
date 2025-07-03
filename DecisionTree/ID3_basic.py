import math
import pandas as pd
from collections import Counter
from graphviz import Digraph


# 计算数据集的熵
def entropy(data, label_col):
    labels = data[label_col]
    total = len(labels)
    if total == 0:
        return 0
    label_counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in label_counts.values() if count > 0)


# 计算某个属性的信息增益
def info_gain(data, attr, label_col):
    total_entropy = entropy(data, label_col)
    if attr in data.columns and pd.api.types.is_numeric_dtype(data[attr]):
        # 处理连续属性
        values = sorted(data[attr])
        split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        best_info = float('inf')
        for split in split_points:
            left = data[data[attr] <= split]
            right = data[data[attr] > split]
            weighted_entropy = (len(left) / len(data)) * entropy(left, label_col) + (len(right) / len(data)) * entropy(
                right, label_col)
            if weighted_entropy < best_info:
                best_info = weighted_entropy
        return total_entropy - best_info
    else:
        # 处理离散属性
        values = data[attr].unique()
        weighted_entropy = 0
        for val in values:
            subset = data[data[attr] == val]
            weighted_entropy += (len(subset) / len(data)) * entropy(subset, label_col)
        return total_entropy - weighted_entropy


# 表示决策树节点的类
class Node:
    def __init__(self, attribute=None, threshold=None, label=None, branches=None):
        self.attribute = attribute  # 分裂属性
        self.threshold = threshold  # 连续属性的阈值
        self.label = label  # 叶节点标签
        self.branches = branches if branches is not None else {}  # 子节点


# ID3算法构建决策树
def id3(data, attributes, label_col):
    labels = data[label_col]
    # 如果所有样本的标签相同，返回叶节点
    if len(labels.unique()) == 1:
        return Node(label=labels.iloc[0])

    # 如果没有可用属性，返回多数类标签的叶节点
    if not attributes:
        return Node(label=Counter(labels).most_common(1)[0][0])

    # 找到最佳分裂属性
    gains = [(attr, info_gain(data, attr, label_col)) for attr in attributes]
    best_attr, _ = max(gains, key=lambda x: x[1])

    # 为最佳属性创建节点
    if pd.api.types.is_numeric_dtype(data[best_attr]):
        # 处理连续属性
        values = sorted(data[best_attr])
        split_points = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        best_info = float('inf')
        best_split = None
        for split in split_points:
            left = data[data[best_attr] <= split]
            right = data[data[best_attr] > split]
            weighted_entropy = (len(left) / len(data)) * entropy(left, label_col) + (len(right) / len(data)) * entropy(
                right, label_col)
            if weighted_entropy < best_info:
                best_info = weighted_entropy
                best_split = split
        node = Node(attribute=best_attr, threshold=best_split)
        left_data = data[data[best_attr] <= best_split]
        right_data = data[data[best_attr] > best_split]
        node.branches['<='] = id3(left_data, attributes - {best_attr}, label_col)
        node.branches['>'] = id3(right_data, attributes - {best_attr}, label_col)
    else:
        # 处理离散属性
        node = Node(attribute=best_attr)
        for val in data[best_attr].unique():
            subset = data[data[best_attr] == val]
            if len(subset) == 0:
                node.branches[val] = Node(label=Counter(labels).most_common(1)[0][0])
            else:
                node.branches[val] = id3(subset, attributes - {best_attr}, label_col)

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


# 可视化决策树
def plot_tree(tree, dot=None, parent=None, edge_label=None):
    if dot is None:
        dot = Digraph(comment='Decision Tree')
        dot.attr(rankdir='TB')  # 从上到下

    # 生成唯一节点ID
    node_id = str(id(tree))

    if tree.label is not None:
        # 叶节点
        dot.node(node_id, f'Class: {tree.label}', shape='box', style='filled', fillcolor='lightgreen')
    else:
        # 决策节点
        if tree.threshold is not None:
            label = f'{tree.attribute} <= {tree.threshold:.2f}'
        else:
            label = f'{tree.attribute}'
        dot.node(node_id, label, shape='ellipse', style='filled', fillcolor='lightblue')

    if parent is not None:
        dot.edge(parent, node_id, label=edge_label)

    # 递归绘制子节点
    for branch_label, subtree in tree.branches.items():
        plot_tree(subtree, dot, node_id, branch_label)

    return dot


# 主函数运行实验
def main():
    # 加载训练和测试数据
    train_data = pd.read_csv('traindata.txt', sep='\s+', header=None,
                             names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])
    test_data = pd.read_csv('testdata.txt', sep='\s+', header=None,
                            names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])

    # 定义属性和标签列
    attributes = set(train_data.columns) - {'label'}
    label_col = 'label'

    # 构建决策树
    tree = id3(train_data, attributes, label_col)

    # 可视化决策树
    dot = plot_tree(tree)
    dot.render('decision_tree', format='png', cleanup=True)  # 保存为 decision_tree.png
    print("决策树已保存为 decision_tree.png")

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