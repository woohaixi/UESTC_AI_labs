import heapq
import time

def get_misplaced_tiles(state, goal_state, n):
    """
    计算当前状态下与目标状态相比位置错误的瓦片数量（忽略空格0）。
    """
    count = 0
    for i in range(n):
        for j in range(n):
            if state[i][j] != goal_state[i][j] and state[i][j] != 0:
                count += 1
    return count

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
    h_start = get_misplaced_tiles(initial_state, goal_state, n)
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
            h_neighbor = get_misplaced_tiles(neighbor_state, goal_state, n)
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