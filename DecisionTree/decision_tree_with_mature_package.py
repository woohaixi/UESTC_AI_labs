import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


# 加载数据（假设数据格式：4个特征 + 1个标签列）
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, sep='\s+', header=None,
                             names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])
    test_data = pd.read_csv(test_file, sep='\s+', header=None,
                            names=['attr1', 'attr2', 'attr3', 'attr4', 'label'])
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    return X_train, y_train, X_test, y_test


# 构建决策树（ID3、CART）并可视化
def build_and_visualize_tree(X_train, y_train, algorithm='CART',
                             max_depth=5, min_samples_split=2, feature_names=None):
    # 创建决策树分类器
    if algorithm == 'ID3':
        clf = DecisionTreeClassifier(criterion='entropy',
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split)
    elif algorithm == 'CART':
        clf = DecisionTreeClassifier(criterion='gini',
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split)
    else:
        raise ValueError("Unknown algorithm. Choose 'ID3' or 'CART'.")

    # 训练模型
    clf.fit(X_train, y_train)

    # 可视化决策树
    plt.figure(figsize=(20, 10))
    plot_tree(clf,
              filled=True,
              feature_names=feature_names,
              class_names=[str(c) for c in clf.classes_],
              rounded=True,
              proportion=True,
              fontsize=10)
    plt.title(f'Decision Tree ({algorithm} Algorithm)', fontsize=14)
    plt.show()

    return clf


# 评估模型
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 主函数
def main():
    # 加载数据
    X_train, y_train, X_test, y_test = load_data('traindata.txt', 'testdata.txt')

    # 获取特征名称（假设数据有4个特征）
    feature_names = X_train.columns.tolist()

    # 构建并评估两种决策树
    algorithms = ['ID3', 'CART']
    results = {}

    for algo in algorithms:
        print(f"\n=== {algo} 算法 ===")
        clf = build_and_visualize_tree(X_train, y_train,
                                       algorithm=algo,
                                       max_depth=5,
                                       min_samples_split=2,
                                       feature_names=feature_names)
        accuracy = evaluate_model(clf, X_test, y_test)
        results[algo] = accuracy
        print(f"{algo} 准确率: {accuracy:.4f}")

    # 输出最佳算法
    best_algo = max(results, key=results.get)
    print(f"\n最佳算法: {best_algo} (准确率: {results[best_algo]:.4f})")


if __name__ == "__main__":
    main()