import heapq
import time
import math


def get_linear_conflict_heuristic(state, goal_state, n):
    """
    计算当前状态的启发值（曼哈顿距离 + 线性冲突修正）
    :param state: 当前状态（二维列表）
    :param goal_state: 目标状态（二维列表）
    :param n: 拼图尺寸（如3表示3x3）
    :return: 启发值（曼哈顿距离 + 2*冲突对数）
    """
    # 1. 计算曼哈顿距离
    manhattan = 0
    # 存储每个数字的目标位置（优化查找）
    goal_pos = {}
    for i in range(n):
        for j in range(n):
            tile = goal_state[i][j]
            if tile != 0:
                goal_pos[tile] = (i, j)

    for i in range(n):
        for j in range(n):
            tile = state[i][j]
            if tile != 0:
                target_i, target_j = goal_pos[tile]
                manhattan += abs(i - target_i) + abs(j - target_j)

    # 2. 计算线性冲突
    conflicts = 0

    # 检查行冲突
    for row in range(n):
        tiles_in_row = []
        # 收集当前行中所有非空且目标也在该行的数字
        for col in range(n):
            tile = state[row][col]
            if tile != 0 and goal_pos[tile][0] == row:
                tiles_in_row.append((col, tile))

        # 检查是否有数字需要交换
        for i in range(len(tiles_in_row)):
            for j in range(i + 1, len(tiles_in_row)):
                col_i, tile_i = tiles_in_row[i]
                col_j, tile_j = tiles_in_row[j]
                # 如果当前顺序与目标顺序相反，则冲突
                if (col_i < col_j and goal_pos[tile_i][1] > goal_pos[tile_j][1]) or \
                        (col_i > col_j and goal_pos[tile_i][1] < goal_pos[tile_j][1]):
                    conflicts += 1

    # 检查列冲突（逻辑与行类似）
    for col in range(n):
        tiles_in_col = []
        for row in range(n):
            tile = state[row][col]
            if tile != 0 and goal_pos[tile][1] == col:
                tiles_in_col.append((row, tile))

        for i in range(len(tiles_in_col)):
            for j in range(i + 1, len(tiles_in_col)):
                row_i, tile_i = tiles_in_col[i]
                row_j, tile_j = tiles_in_col[j]
                if (row_i < row_j and goal_pos[tile_i][0] > goal_pos[tile_j][0]) or \
                        (row_i > row_j and goal_pos[tile_i][0] < goal_pos[tile_j][0]):
                    conflicts += 1

    return manhattan + 2 * conflicts

def get_neighbors(state, n):
    """
    获取当前状态的所有可能邻居状态。
    """
    neighbors = []
    for i in range(n):
        for j in range(n):
            if state[i][j] == 0:
                zero_row, zero_col = i, j
                break
        else:
            continue
        break

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for move_row, move_col in moves:
        new_row, new_col = zero_row + move_row, zero_col + move_col
        if 0 <= new_row < n and 0 <= new_col < n:
            new_state = [list(row) for row in state]
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

    open_list = []
    closed_list = {}
    g_start = 0
    h_start = get_linear_conflict_heuristic(initial_state, goal_state, n)
    f_start = g_start + h_start

    heapq.heappush(open_list, (f_start, g_start, initial_state_tuple, None))
    closed_list[initial_state_tuple] = (g_start, None)

    while open_list:
        f_current, g_current, current_state_tuple, parent_state_tuple = heapq.heappop(open_list)
        current_state = [list(row) for row in current_state_tuple]

        if current_state_tuple == goal_state_tuple:
            print(f"找到解！总步数: {g_current}")
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
                print("-" * (n * 4))
            return g_current

        neighbors = get_neighbors(current_state, n)
        for neighbor_state in neighbors:
            neighbor_state_tuple = tuple(tuple(row) for row in neighbor_state)
            g_neighbor = g_current + 1
            h_neighbor = get_linear_conflict_heuristic(neighbor_state, goal_state, n)
            f_neighbor = g_neighbor + h_neighbor

            if neighbor_state_tuple in closed_list and g_neighbor >= closed_list[neighbor_state_tuple][0]:
                continue

            found_in_open = False
            for i, (f, g, state, parent) in enumerate(open_list):
                if state == neighbor_state_tuple:
                    found_in_open = True
                    if g_neighbor < g:
                        open_list[i] = (f_neighbor, g_neighbor, neighbor_state_tuple, current_state_tuple)
                        heapq.heapify(open_list)
                        closed_list[neighbor_state_tuple] = (g_neighbor, current_state_tuple)
                    break

            if not found_in_open:
                heapq.heappush(open_list, (f_neighbor, g_neighbor, neighbor_state_tuple, current_state_tuple))
                closed_list[neighbor_state_tuple] = (g_neighbor, current_state_tuple)

    print("无解")
    return -1

def get_matrix_input_by_row(size, prompt):
    """
    获取用户按行输入的矩阵。
    """
    print(prompt)
    matrix = []
    for i in range(size):
        while True:
            try:
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
    n = 3
    if len(matrix) != n or any(len(row) != n for row in matrix):
        print("内部错误：矩阵不是 3x3 的。")
        return False

    flat_list = [item for sublist in matrix for item in sublist]
    expected_numbers = list(range(9))
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

def is_solvable_8(initial_state, goal_state):
    """
    判断八数码问题是否有解。
    对于N=3（奇数），当且仅当逆序数相等时有解。
    """
    initial_inversion = get_inversion_count(initial_state)
    goal_inversion = get_inversion_count(goal_state)
    return initial_inversion % 2 == goal_inversion % 2

if __name__ == "__main__":
    n_value = 3

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

    if not is_solvable_8(initial_matrix, goal_matrix):
        print("该八数码问题无解。")
    else:
        print("该八数码问题有解，开始求解...")
        solve_n_puzzle(initial_matrix, goal_matrix)