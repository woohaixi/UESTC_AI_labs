import heapq
import time

def get_manhattan_distance(state, goal_state, n):
    """
    计算当前状态下所有数码块的曼哈顿距离之和。
    """
    distance = 0
    for i in range(n):
        for j in range(n):
            tile = state[i][j]
            if tile != 0:
                # 找到该数码块在目标状态中的位置
                for target_i in range(n):
                    for target_j in range(n):
                        if goal_state[target_i][target_j] == tile:
                            distance += abs(i - target_i) + abs(j - target_j)
                            break
                    else:
                        continue # 继续内层循环
                    break # 退出外层循环
    return distance

def get_neighbors(state, n):
    """
    获取当前状态的所有可能邻居状态。
    """
    neighbors = []
    # 找到空格的位置
    for i in range(n):
        for j in range(n):
            if state[i][j] == 0:
                zero_row, zero_col = i, j
                break
        else:
            continue
        break

    # 可能的移动方向：上、下、左、右
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for move_row, move_col in moves:
        new_row, new_col = zero_row + move_row, zero_col + move_col

        # 检查移动是否在边界内
        if 0 <= new_row < n and 0 <= new_col < n:
            # 创建新的状态（复制当前状态）
            new_state = [list(row) for row in state]
            # 交换空格和相邻的数码块
            new_state[zero_row][zero_col] = new_state[new_row][new_col]
            new_state[new_row][new_col] = 0
            neighbors.append(new_state)

    return neighbors

def solve_n_puzzle(initial_state, goal_state):
    """
    使用A*算法求解N数码问题。
    """
    n = len(initial_state)
    initial_state_tuple = tuple(tuple(row) for row in initial_state)
    goal_state_tuple = tuple(tuple(row) for row in goal_state)

    # 优先队列存储待考察的节点 (f_value, g_value, state_tuple, parent_state_tuple)
    # f_value 用于排序
    open_list = []
    # 存储已考察的节点及其信息 (state_tuple: (g_value, parent_state_tuple))
    closed_list = {}

    # 起始节点的 g 值为 0
    g_start = 0
    # 起始节点的 h 值
    h_start = get_manhattan_distance(initial_state, goal_state, n)
    # 起始节点的 f 值
    f_start = g_start + h_start

    # 将起始节点加入开放列表
    heapq.heappush(open_list, (f_start, g_start, initial_state_tuple, None))
    # 将起始节点加入封闭列表，记录其 g 值和父节点（None）
    closed_list[initial_state_tuple] = (g_start, None)

    while open_list:
        # 从开放列表中取出 f 值最小的节点
        f_current, g_current, current_state_tuple, parent_state_tuple = heapq.heappop(open_list)
        current_state = [list(row) for row in current_state_tuple]

        # 如果当前节点是目标节点，则找到解
        if current_state_tuple == goal_state_tuple:
            print(f"找到解！总步数: {g_current}")
            # 回溯路径并打印移动过程
            path = []
            step = current_state_tuple
            while step is not None:
                path.append(step)
                step = closed_list[step][1]
            path.reverse()

            print("移动过程:")
            for i, state_tuple in enumerate(path):
                print(f"步骤 {i}:")
                for row in state_tuple:
                    print(row)
                print("-" * (n * 4)) # 分隔线

            return g_current # 返回步数

        # 获取当前状态的邻居状态
        neighbors = get_neighbors(current_state, n)

        for neighbor_state in neighbors:
            neighbor_state_tuple = tuple(tuple(row) for row in neighbor_state)
            # 到邻居节点的 g 值为当前节点的 g 值 + 1
            g_neighbor = g_current + 1
            # 计算邻居节点的 h 值
            h_neighbor = get_manhattan_distance(neighbor_state, goal_state, n)
            # 计算邻居节点的 f 值
            f_neighbor = g_neighbor + h_neighbor

            # 如果邻居节点已在封闭列表中，并且新的 g 值不比之前的小，则跳过
            if neighbor_state_tuple in closed_list and g_neighbor >= closed_list[neighbor_state_tuple][0]:
                continue

            # 如果邻居节点已在开放列表中，并且新的 g 值比之前的小，则更新开放列表中的信息
            found_in_open = False
            for i, (f, g, state, parent) in enumerate(open_list):
                if state == neighbor_state_tuple:
                    found_in_open = True
                    if g_neighbor < g:
                        # 更新信息并重新组织堆
                        open_list[i] = (f_neighbor, g_neighbor, neighbor_state_tuple, current_state_tuple)
                        heapq.heapify(open_list) # 重新堆化
                        # 更新封闭列表中的信息（以防万一，尽管应该在open list中更新）
                        closed_list[neighbor_state_tuple] = (g_neighbor, current_state_tuple)
                    break

            # 如果邻居节点既不在开放列表中也不在封闭列表中，则加入开放列表
            if not found_in_open:
                heapq.heappush(open_list, (f_neighbor, g_neighbor, neighbor_state_tuple, current_state_tuple))
                # 加入封闭列表，记录其 g 值和父节点
                closed_list[neighbor_state_tuple] = (g_neighbor, current_state_tuple)

    # 如果开放列表为空且未找到解，则无解
    print("无解")
    return -1 # 返回 -1 表示无解

# --- 用户输入部分 (支持按行输入，无行号提示) ---

def get_matrix_input_by_row(size, prompt):
    """
    获取用户按行输入的矩阵。
    """
    print(prompt)
    matrix = []
    for i in range(size):
        while True:
            try:
                # 去掉行号提示
                row_input = input().split()
                if len(row_input) != size:
                    print(f"输入错误：每行需要 {size} 个数字。请重新输入。")
                    continue
                row = [int(x) for x in row_input]
                matrix.append(row)
                break
            except ValueError:
                print("输入错误：请确保输入的是整数。请重新输入。")
    return matrix

def is_valid_puzzle_8(matrix):
    """
    检查输入的矩阵是否是有效的八数码问题状态。
    """
    n = 3 # 固定为八数码
    # 这部分检查理论上在 get_matrix_input_by_row 中已经完成了尺寸检查，但保留更严谨
    if len(matrix) != n or any(len(row) != n for row in matrix):
        print("内部错误：矩阵不是 3x3 的。")
        return False

    flat_list = [item for sublist in matrix for item in sublist]
    expected_numbers = list(range(9)) # 八数码包含0到8
    if sorted(flat_list) != expected_numbers:
        print("输入错误：八数码矩阵必须包含从 0 到 8 的所有数字。")
        return False
    return True

def get_inversion_count(matrix):
    """
    计算矩阵的逆序数。
    """
    flat_list = [item for sublist in matrix for item in sublist if item != 0]
    inversion_count = 0
    n = len(flat_list)
    for i in range(n):
        for j in range(i + 1, n):
            if flat_list[i] > flat_list[j]:
                inversion_count += 1
    return inversion_count

# 对于八数码（N=3，奇数），只需判断逆序数的奇偶性是否相同
def is_solvable_8(initial_state, goal_state):
    """
    判断八数码问题是否有解。
    对于N=3（奇数），当且仅当逆序数相等时有解。
    """
    initial_inversion = get_inversion_count(initial_state)
    goal_inversion = get_inversion_count(goal_state)

    return initial_inversion % 2 == goal_inversion % 2

if __name__ == "__main__":
    n_value = 3 # 固定为八数码

    while True:
        initial_matrix = get_matrix_input_by_row(n_value, "请输入初始矩阵（每行3个数字，用空格分隔）：")
        if is_valid_puzzle_8(initial_matrix):
            break
        else:
            print("初始矩阵输入无效，请重新输入。")

    while True:
        goal_matrix = get_matrix_input_by_row(n_value, "请输入目标矩阵（每行3个数字，用空格分隔）：")
        if is_valid_puzzle_8(goal_matrix):
            break
        else:
            print("目标矩阵输入无效，请重新输入。")

    # 检查是否有解
    if not is_solvable_8(initial_matrix, goal_matrix):
        print("该八数码问题无解。")
    else:
        print("该八数码问题有解，开始求解...")
        solve_n_puzzle(initial_matrix, goal_matrix) # 调用通用的求解函数
