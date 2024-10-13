from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot
from matplotlib import style
import pandas as pd
import heapq

#可以设置一个终极奖励，当两个泄漏点都被探测到的时候 顺便解算最后的步数奖励

style.use('ggplot')

# size = 10 #所有的对象都在一个10X10的格子里面
#重新将数据整理为15X15的范围 因此修改size为15

# #导入点浓度数据
# # 指定Excel文件的路径
# file_path = r"C:\Users\a\Desktop\工作文件夹\文献投稿\嗅觉传感器2.0\两点数据.xlsx"
# # 读取Excel文件中的数据
# data = pd.read_excel(file_path)
# data_array = data.values
# # 查看前几行数据  ctrl+/实现快速注释
# #print(data.head())

# 定义参数
# x0_1, y0_1 = 32, 48  # 第一个泄漏点坐标
# x0_2, y0_2 = 80, 30  # 第二个泄漏点坐标

Q = 300000  # 泄漏量
u = 1  # 风速
h = 1  # 泄漏点高度
H = 1  # 大气稳定度参数
tx = 20  # 水平扩散系数
ty = 5  # 水平扩散系数
tz = 5  # 垂直扩散系数

# 设定网格坐标
X, Y = np.meshgrid(np.arange(0, 101, 1), np.arange(0, 81, 1))

# 初始化浓度矩阵
C0 = np.zeros_like(X, dtype=float)

# 计算第一个泄漏点的污染物质浓度分布
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xi = X[i, j]  # 网格点横坐标
        yi = Y[i, j]  # 网格点纵坐标

        C0[i, j] += Q / (2 * np.pi) ** 1.5 / u / tx / ty / tz * \
                    np.exp(-(xi - x0_1) ** 2 / (2 * tx ** 2)) * \
                    np.exp(-(yi - y0_1) ** 2 / (2 * ty ** 2)) * \
                    (np.exp(-h ** 2 / (2 * tz ** 2)) + np.exp(-4 * H ** 2 / (2 * tz ** 2)))

# 计算第二个泄漏点的污染物质浓度分布
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xi = X[i, j]  # 网格点横坐标
        yi = Y[i, j]  # 网格点纵坐标

        C0[i, j] += Q / (2 * np.pi) ** 1.5 / u / tx / ty / tz * \
                    np.exp(-(xi - x0_2) ** 2 / (2 * tx ** 2)) * \
                    np.exp(-(yi - y0_2) ** 2 / (2 * ty ** 2)) * \
                    (np.exp(-h ** 2 / (2 * tz ** 2)) + np.exp(-4 * H ** 2 / (2 * tz ** 2)))

# 将生成的数据命名为 data
data = C0
# 将 data 中所有小于 1 的值替换为 0
data[data < 1 ] = 0

# 展示数据
# plt.figure(figsize=(14, 7))
# plt.imshow(data, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='viridis', aspect='auto')
# plt.colorbar(label='Concentration')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Concentration Distribution with Two Leak Points')
# plt.show()

EPISODES = 30000 #进行训练的智能体 要进行的游戏局数
SHOW_EVERY = 5000#每个3000局 展示一次训练结果
epsilon = 0.50 #在运动情况下 50%的情况会进行随机的运动  剩余的50%是累计下来的上局经验
step_max = 600
#为了初期更全面的探索 暂时增加值 使得更随机
#可以再根据该点的数据值 例如 大于50 才继续走相同的路 不一定只是浓度梯度

EPS_DECAY = 0.9999#用于衰减这个随机性  因为玩的越多 经验越足 更快走到目的地
DISCOUNT = 0.95#折扣汇报  可能这次不是最优解  暂时降低这个回报
LEARNING_RATE = 0.1#学习速率

#可以将之前运行过得QTable用于这里训练
q_table = None

#增加一个环境的类
class envCube:#包含环境的状态、奖励机制、动作空间以及渲染方法等
    width = 101  # 对应 X 的范围 0 到 108
    height = 81  # 对应 Y 的范围 -40 到 40，总共 81 个点
    OBSERVATION_SPACE_VALUES = (width, height, 4)
    ACTION_SPACE_VALUES = 9  # 保持不变
    return_image = False

    #设置固定位置
    player_start_pos = (0, 0)  # 玩家起始位置
    # enemy_positions = [(1, 8), (50, 13), (80, 11)]  # 多个敌人位置
    enemy_positions = [(80, 11)]  # 多个敌人位置
    enemies = []# 初始化敌人列表
    # enemy_positions = [(1, 8)]  # 多个敌人位置
    foods = []#初始化食物位置
    def generate_random_target(self, center):
        x, y = center
        return (np.random.randint(x - 2, x + 3), np.random.randint(y - 2, y + 3))

    # 新增的属性
    position_visit_counts = {}  # 记录每个位置被访问的次数
    REVISIT_PENALTY_FACTOR = 10  # 每次重复访问时增加的惩罚系数
    recent_actions = []  # 用于存储最近的动作 防止原地打转
    low_concentration_counts = {}  # 记录低浓度位置的访问次数
    LOW_CONCENTRATION_THRESHOLD = 10#低浓度阈值
    LOW_CONCENTRATION_PENALTY = -5#低浓度惩罚
    LOW_CONCENTRATION_PENALTY_FACTOR = 1.5

    # 找到浓度最大的两个点的位置
    flat_data = data.flatten()
    sorted_indices = np.argsort(flat_data)[-2:]  # 找到最大值的索引
    positions = np.unravel_index(sorted_indices, data.shape)  # 将平面索引转换为矩阵索引
    food_positions = list(zip(positions[1], positions[0]))  # 转换为列表形式，注意 x 和 y 的顺序

    # interest_points = [(24,40),(70,20)]#关键点第1组
    # current_interest_point = None # None 表示还未到达兴趣点，0 表示第一个目标，1 表示第二个目标
    # reached_interest_point = False
    #
    # close_points = [(40,55),(90,40)]#关键点2
    # current_close_point = None

    # 设置奖励与惩罚的机制
    FOOD_REWARD = 500  # 吃到食物之后的奖励（获取到最大浓度点的时候的奖励  这里不能设置太小 不然吃到食物奖励还没有刷浓度步数的奖励高
    ENEMY_PENALITY = -300  # 被敌人逮住扣300  （触碰到行走的测量边界  或者说低浓度区域
    MOVE_PENALITY = -1  # 每移动一次 扣一分 争取最短的路径走到目的地  （走到浓度最大的区域
    LIMIT_PENALITY = -2 #移动到边界上 就要扣分 （防止边界上来回运动
    Gradient_reward = 1 #每朝着一次浓度变大的方向运动 获得一次小奖励
    Gradient_1_reward = 1  # 当探测到的浓度大于20时 获得奖励
    Gradient_2_reward = 2 #当探测到的浓度大于40时 获得奖励
    Gradient_3_reward = 3  # 当探测到的浓度大于60时 获得奖励
    Gradient_4_reward = 4  # 当探测到的浓度大于70时 获得奖励
    current_target = 0  # 0 表示第一个目标，1 表示第二个目标
    visited_positions = set()#获取位置
    REVISIT_PENALTY = -20  # 若长时间待在这里 给予惩罚
    steps_taken = 0  # 记录已走的步数

    current_stage = 0  # 0: 初始阶段, 1: 第一个泄漏点探测, 2: 第二个泄漏点探测
    leak_points = [(32, 48), (80, 30)]
    current_target = None
    initial_phase = True  # 用于标记是否在初始阶段
    initial_area_reached = False  # 用于标记是否已到达初始区域
    detected_leaks = []
    exploration_targets = []  # We'll initialize this in the reset method

    # 设置图像的颜色
    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)}  # red
    # 设置颜色
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    player_trajectory = deque(maxlen=100)  # 存储最近50步的轨迹
    total_score = 0  # 总分
    collected_food = [False, False]  # 跟踪哪些食物点已被采集
    q_table = {}  # 初始化为空字典

    detected_points = []
    current_target = None
    recent_concentrations = deque(maxlen=100)

    initial_target = (32, 48)
    initial_phase = True  # 用于标记是否在初始阶段
    initial_area_reached = False  # 用于标记是否已到达初始区域

    def reset(self):
        self.player = Cube(self.width, self.height, x=0, y=0)
        self.initial_phase = True
        self.initial_area_reached = False

        self.current_stage = 0
        self.detected_leaks = []

        # Initialize exploration_targets here
        self.exploration_targets = [
            self.generate_random_target(self.leak_points[0]),
            self.generate_random_target(self.leak_points[1])
        ]

        if self.exploration_targets:
            self.current_target = self.exploration_targets[0]
        else:
            print("Warning: exploration_targets is empty. Setting current_target to None.")
            self.current_target = None

        # Initialize enemies as Cube objects
        self.enemies = [Cube(self.width, self.height, x=pos[0], y=pos[1]) for pos in self.enemy_positions]

        # Reset other necessary attributes
        self.episode_step = 0
        self.total_score = 0
        self.player_trajectory = deque(maxlen=50)
        self.recent_concentrations = deque(maxlen=100)
        self.position_visit_counts = {}
        self.low_concentration_counts = {}
        self.visited_positions = set()

        # 初始化两个目标（泄漏点）
        self.foods = []
        for pos in self.food_positions:
            x, y = pos
            if 0 <= x < self.width and 0 <= y < self.height:
                food = Cube(self.width, self.height, x=x, y=y)
                self.foods.append(food)
            else:
                print(f"警告: 食物位置 {x}, {y} 超出环境边界，使用随机位置替代。")
                food = Cube(self.width, self.height)
                self.foods.append(food)

        # 确保有两个目标
        while len(self.foods) < 2:
            food = Cube(self.width, self.height)
            self.foods.append(food)

        self.current_target = 0
        self.collected_food = [False, False]

        self.position_visit_counts = {}  # 重置访问次数记录

        # 初始化敌人
        self.enemies = []
        for pos in self.enemy_positions:
            enemy = Cube(self.width, self.height, x=pos[0], y=pos[1])
            self.enemies.append(enemy)

        # 确保玩家不与任何食物或敌人重叠
        while any(self.player == food for food in self.foods) or any(self.player == enemy for enemy in self.enemies):
            self.player = Cube(self.width, self.height)

        # 记录踪迹和总分
        self.player_trajectory.clear()
        self.total_score = 0
        self.collected_food = [False for _ in self.foods]  # 记录点是否被采集

        # 设置初始目标
        self.collected_food = [False for _ in self.food_positions]  # 确保长度匹配
        self.current_target = 0
        # 记录位置
        self.visited_positions.clear()

        # 重置步数计数器
        self.episode_step = 0

        # 重置低浓度计数
        self.low_concentration_counts = {}
        # 创建观察空间
        observation = self.get_observation()

        #记录探测的数据 找到最大探测点
        self.detected_points = []
        self.current_target = None
        self.recent_concentrations = deque(maxlen=100)

        self.detected_leaks = []
        self.first_detection_reward = None
        self.first_detection_episode = None
        self.second_detection_reward = None
        self.second_detection_episode = None
        self.current_target = None
        return self.get_observation()

    def get_observation(self):
        if len(self.detected_points) == 0:
            return (
                self.player.x,
                self.player.y,
                0,  # 如果没有探测到点，设置目标坐标差为 0
                0,
                self.player.x - self.enemies[0].x if self.enemies else 0,
                self.player.y - self.enemies[0].y if self.enemies else 0,
                0  # 0 表示没有当前目标
            )
        elif len(self.detected_points) == 1:
            return (
                self.player.x,
                self.player.y,
                self.player.x - self.detected_points[0][0],
                self.player.y - self.detected_points[0][1],
                self.player.x - self.enemies[0].x if self.enemies else 0,
                self.player.y - self.enemies[0].y if self.enemies else 0,
                1  # 1 表示有一个探测点
            )
        else:
            return (
                self.player.x,
                self.player.y,
                self.player.x - self.detected_points[0][0],
                self.player.y - self.detected_points[0][1],
                self.player.x - self.enemies[0].x if self.enemies else 0,
                self.player.y - self.enemies[0].y if self.enemies else 0,
                2  # 2 表示有两个探测点
            )
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def astar(self, start, goal):
        heap = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}

        while heap:
            current = heapq.heappop(heap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < self.width and
                        0 <= neighbor[1] < self.height):

                    tentative_g_score = g_score[current] + 1

                    if (neighbor not in g_score or
                            tentative_g_score < g_score[neighbor]):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = (g_score[neighbor] +
                                             self.manhattan_distance(neighbor, goal))
                        heapq.heappush(heap, (f_score[neighbor], neighbor))

        return None  # 没有找到路径

    def step(self, action):
        self.episode_step += 1  # 记录走了多少步
        done = False  # 初始化 done 为 False
        reward = 0
        if self.initial_phase:
            # 在初始阶段，直接移动towards目标
            if self.current_target:
                target_x, target_y = self.current_target
                dx = np.sign(target_x - self.player.x)
                dy = np.sign(target_y - self.player.y)
                self.player.move(x=dx, y=dy)

                # 检查是否到达目标区域
                if self.manhattan_distance((self.player.x, self.player.y), self.current_target) <= 1:
                    self.initial_phase = False
                    self.initial_area_reached = True
                    print("到达初始探索区域！开始正常探测。")
            else:
                print("Warning: current_target is None in initial phase.")

            # 在初始阶段，reward保持为0
            reward = 0

            # 执行动作
        self.player.action(action)
        current_position = (self.player.x, self.player.y)# 保存当前位置
        x1, y1 = current_position

        # 确保玩家在边界内
        self.player.x = max(0, min(self.player.x, self.width - 1))
        self.player.y = max(0, min(self.player.y, self.height - 1))
        # 保存移动后的位置
        new_position = (self.player.x, self.player.y)
        x2, y2 = new_position

        # 确保 x1, y1, x2 和 y2 不超出 data 的范围
        x1 = min(max(x1, 0), data.shape[0] - 1)
        y1 = min(max(y1, 0), data.shape[1] - 1)
        x2 = min(max(x2, 0), data.shape[0] - 1)
        y2 = min(max(y2, 0), data.shape[1] - 1)

        # 计算浓度梯度
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        nongdu_current = data[min(x1, data.shape[0] - 1), min(y1, data.shape[1] - 1)]
        nongdu_previous = data[min(x2, data.shape[0] - 1), min(y2, data.shape[1] - 1)]
        gradient = (nongdu_current - nongdu_previous) / distance if distance != 0 else 0

        # 根据梯度调整方向
        if gradient > 0:
            # 梯度大于0时，继续沿着当前方向行进
            if nongdu_current > 20:
                self.player.action(action)
            else:
                if np.random.rand() < 0.95:
                    self.player.action(action)  # 95%的概率沿着当前方向走
                else:
                    action = np.random.randint(self.ACTION_SPACE_VALUES)
                    self.player.action(action)  # 5%的概率随机选择其他方向
        else:
            # 梯度小于0时，从剩余的方向中随机选择一个新的方向
            available_actions = list(range(self.ACTION_SPACE_VALUES))
            available_actions.remove(action)
            action = np.random.choice(available_actions)
            self.player.action(action)

        # 检查是否碰到敌人
        if any(self.player == enemy for enemy in self.enemies):
            reward = self.ENEMY_PENALITY
            done = True

        # 获取当前位置的浓度
        current_concentration = data[self.player.y, self.player.x]
        self.recent_concentrations.append(current_concentration)

        # 检查是否发现泄漏点
        current_pos = (self.player.x, self.player.y)
        for leak_point in self.leak_points:
            if self.manhattan_distance(current_pos, leak_point) <= 3 and leak_point not in self.detected_leaks:
                self.detected_leaks.append(leak_point)
                if len(self.detected_leaks) == 1:
                    self.first_detection_reward = self.total_score
                    self.first_detection_episode = self.episode_step
                    print(f"探测到第一个泄漏点！位置: {leak_point}, 时刻: {self.episode_step}, 奖励: {self.total_score}")
                elif len(self.detected_leaks) == 2:
                    self.second_detection_reward = self.total_score
                    self.second_detection_episode = self.episode_step
                    print(f"探测到第二个泄漏点！位置: {leak_point}, 时刻: {self.episode_step}, 奖励: {self.total_score}")
                reward += self.FOOD_REWARD
                done = True  # 发现泄漏点后结束当前回合

        # 如果有目标位置，检查是否到达
        if self.current_target:
            if self.player.x == self.current_target[0] and self.player.y == self.current_target[1]:
                reward += self.FOOD_REWARD
                done = True

        # 记录最近的动作
        self.recent_actions.append(action)
        if len(self.recent_actions) > 2:
            self.recent_actions.pop(0)

        #玩家与敌人的位置
        if any(self.player == enemy for enemy in self.enemies):
            reward = self.ENEMY_PENALITY
            done = True

            # 如果新位置超出边界或者动作重复，选择新的方向
            if (self.player.x < 0 or self.player.x >= self.width or
                    self.player.y < 0 or self.player.y >= self.height or
                    action in self.recent_actions):

                available_actions = list(range(self.ACTION_SPACE_VALUES))
                for recent_action in self.recent_actions:
                    if recent_action in available_actions:
                        available_actions.remove(recent_action)

                if available_actions:
                    action = np.random.choice(available_actions)
                    self.player.action(action)
                else:
                    # 如果所有动作都被移除，重置为原始位置
                    self.player.x, self.player.y = current_position

        #探测奖励
        else:
            if gradient >= 0:
                if nongdu_current >= 20:
                    reward = self.Gradient_reward + self.Gradient_1_reward # 当探测到的浓度大于20时 获得奖励
                elif nongdu_current >= 40:
                    reward = self.Gradient_reward + self.Gradient_2_reward # 当探测到的浓度大于20时 获得奖励
                elif nongdu_current >= 60:
                    reward = self.Gradient_reward + self.Gradient_3_reward# 当探测到的浓度大于60时 获得奖励
                elif nongdu_current >= 70:
                    reward = self.Gradient_reward + self.Gradient_4_reward# 当探测到的浓度大于70时 获得奖励
                else:
                    reward = self.Gradient_reward
            else:
                reward = self.MOVE_PENALITY

        # 检查是否达到最大步数
        if self.episode_step >= step_max:
            done = True

            n = 3
            # 若长时间待在这里 给予惩罚
            current_position = (self.player.x, self.player.y)
            if current_position in self.position_visit_counts:
                self.position_visit_counts[current_position] += 2
                revisit_penalty = self.REVISIT_PENALTY * ((self.position_visit_counts[current_position] - 2) ** n) * self.REVISIT_PENALTY_FACTOR
                reward -= revisit_penalty
            else:
                self.position_visit_counts[current_position] = 1

            # 将当前位置添加到已访问集合中
            self.visited_positions.add(current_position)

        # 检查是否是低浓度区域
        if current_concentration < self.LOW_CONCENTRATION_THRESHOLD:
            current_position = (self.player.x, self.player.y)
            if current_position in self.low_concentration_counts:
                self.low_concentration_counts[current_position] += 1
                # 当次数大于3时，给予惩罚
                if self.low_concentration_counts[current_position] > 3:
                    penalty = self.LOW_CONCENTRATION_PENALTY * (self.low_concentration_counts[
                                                                    current_position] ** self.LOW_CONCENTRATION_PENALTY_FACTOR)
                    reward += penalty
                else:
                    reward += 0
            else:
                self.low_concentration_counts[current_position] = 1

        # 更新轨迹和总分
        self.player_trajectory.append((self.player.x, self.player.y))
        self.total_score += reward

        # 获取新观察状态
        new_observation = self.get_observation()

        return self.get_observation(), reward, done

    #降低泄漏点附近的奖励值
    def reduce_nearby_rewards(self, food_index):
        food = self.foods[food_index]
        data_height, data_width = data.shape
        for x in range(max(0, food.x - 2), min(data_width, food.x + 3)):
            for y in range(max(0, food.y - 2), min(data_height, food.y + 3)):
                if 0 <= x < data_width and 0 <= y < data_height:
                    data[y, x] *= -0.1  # 将周围区域的浓度值降低

    def set_negative_concentration(self, x, y):
        for i in range(max(0, x - 5), min(self.width, x + 6)):
            for j in range(max(0, y - 5), min(self.height, y + 6)):
                if data[j, i] > 0:
                    data[j, i] = -data[j, i]

    #画面处理
    def render(self):
        img = self.get_image()
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (800, 640))

        # 绘制已探测的点
        for i, point in enumerate(self.detected_points):
            x, y = point
            x_pos = int(x * 800 / self.width)
            y_pos = int((self.height - 1 - y) * 640 / self.height)
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 第一个点绿色，第二个点红色
            cv2.circle(img, (x_pos, y_pos), 10, color, -1)
            cv2.putText(img, f"D{i + 1}", (x_pos - 10, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制玩家位置
        player_x = int(self.player.x * 800 / self.width)
        player_y = int((self.height - 1 - self.player.y) * 640 / self.height)
        cv2.circle(img, (player_x, player_y), 5, (255, 255, 255), -1)

        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (800, 640))  # 使用 cv2.resize 替代 PIL 的 resize

        cv2.imshow('predator', img)
        cv2.waitKey(1)

        # 添加文本信息
        cv2.putText(img, f"Detected Points: {len(self.detected_points)}/2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Score: {self.total_score:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Steps: {self.episode_step}/{step_max}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # 绘制玩家轨迹
        for i in range(1, len(self.player_trajectory)):
            start = self.player_trajectory[i - 1]
            end = self.player_trajectory[i]
            start_point = (int(start[0] * 800 / self.width), int((self.height - 1 - start[1]) * 640 / self.height))
            end_point = (int(end[0] * 800 / self.width), int((self.height - 1 - end[1]) * 640 / self.height))

            t = i / len(self.player_trajectory)
            r = int(255 * (1 - t))
            g = int(255 * t)
            b = int(128)
            color = (b, g, r)  # BGR格式

            cv2.line(img, start_point, end_point, color, 2)

        cv2.imshow('Environment', img)
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 设置玩家位置
        env[self.height - 1 - self.player.y][self.player.x] = self.d[self.PLAYER_N]

        # 设置所有食物位置
        for i, food in enumerate(self.foods):
            y = self.height - 1 - food.y  # 注意这里的y坐标转换
            x = food.x
            if 0 <= y < self.height and 0 <= x < self.width:  # 确保坐标在有效范围内
                if self.collected_food[i]:
                    env[y][x] = (128, 128, 128)  # 灰色
                elif i == self.current_target:
                    env[y][x] = tuple(min(c * 1.5, 255) for c in self.d[self.FOOD_N])
                else:
                    env[y][x] = self.d[self.FOOD_N]
            # print(f"Rendering Food {i} at: {x}, {y} (original: {food.x}, {food.y})")

        # 设置所有敌人位置
        for enemy in self.enemies:
            env[self.height - 1 - enemy.y][enemy.x] = self.d[self.ENEMY_N]

        return env

    #获取q_table函数
    def get_q_table(self, q_table_name=None):
        if q_table_name is None:
            return self.q_table  # Return the already initialized empty dictionary
        else:
            # If a name is provided, load the Q-table from file
            with open(q_table_name, 'rb') as f:
                self.q_table = pickle.load(f)
            return self.q_table

    def get_q_value(self, state, action):
        if state not in self.q_table:
            # 如果状态不在Q表中，为该状态初始化所有动作的Q值
            self.q_table[state] = [np.random.uniform(-5, 0) for _ in range(self.ACTION_SPACE_VALUES)]
        return self.q_table[state][action]

    def update_q_value(self, state, action, new_value):
        if state not in self.q_table:
            self.q_table[state] = [np.random.uniform(-5, 0) for _ in range(self.ACTION_SPACE_VALUES)]
        self.q_table[state][action] = new_value

class Cube:#玩家、食物或敌人的位置、动作、移动等基本功能。
    def __init__(self, width, height, x=None, y=None):  # 设置位置
        self.width = width
        self.height = height
        self.x = x if x is not None else np.random.randint(0, width)
        self.y = y if y is not None else np.random.randint(0, height)


    def __str__(self):#返还位置
        return f'({self.x},{self.y})'

    def __sub__(self, other):#两个物品之间的位置
        return (self.x-other.x ,self.y-other.y)

    def __eq__(self,other):#如果两者位置一样
        return self.x == other.x and self.y == other.y
    #计算对应的浓度梯度  #需要先前一步的位置和现在的位置

    def action(self,choise):#采取哪一部分动作 上下左右？
        if choise == 0 :#第一个动作
            self.move(x=1,y=1)#向右移动一格 向上移动一格
        elif choise == 1 :#第二个动作
            self.move(x=-1,y=1)#向左移动一格 向上移动一格
        elif choise == 2:  # 第三个动作
            self.move(x=1, y=-1)  #
        elif choise == 3:  # 第四个动作
            self.move(x=-1, y=-1)  #
        elif choise == 4:  # 第五个动作
            self.move(x=0, y=1)  #横着走
        elif choise == 5:  # 第六个动作
            self.move(x=0, y=-1)  #
        elif choise == 6:  # 第7个动作
            self.move(x=-1, y=0)  #
        elif choise == 7:  # 第8个动作
            self.move(x=1, y=0)  #
        elif choise == 8:  # 第9个动作
            self.move(x=0, y=0)  #


    def move(self,x=False,y=False):
        if not x:#如果没有给出下一步怎么移动 则随机动一下
            self.x += np.random.randint(-1,2)#取得值是-1到1之间
        else:
            self.x += x
        if not y:#如果没有给出下一步怎么移动 则随机动一下
            self.y += np.random.randint(-1,2)#取得值是-1到1之间
        else:
                self.y += y
        # 边界检查
        self.x = max(0, min(self.x, self.width - 1))
        self.y = max(0, min(self.y, self.height - 1))
            #设定边界 不让其跑出去
            # if self.x < 0:#小边界
    #         self.x = 0
    #     if self.x >= self.size:#大边界
    #         self.x = self.size-1
    #
    #     if self.y < 0:#小边界
    #         self.y = 0
    #     if self.y >= self.size:#大边界
    #         self.y = self.size-1

    def penalize(self):
        # 惩罚机制，例如减少奖励或增加惩罚
        envCube.reward -= 1  # 假设有一个奖励系统，可以减少奖励

#使用循环进行学习
# 使用循环进行学习
env = envCube()
q_table = env.get_q_table()
all_episode_rewards = []
step_rewards = []  # 用于记录每个步骤的奖励
first_detection = None
second_detection = None

episode_numbers = []
episode_rewards = []
episode_steps = []

# 修改主训练循环
for episode in range(EPISODES):
    obs = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax([env.get_q_value(obs, a) for a in range(env.ACTION_SPACE_VALUES)])
        else:
            action = np.random.randint(0, env.ACTION_SPACE_VALUES)

        new_obs, reward, done = env.step(action)
        steps += 1

        # 更新Q值
        old_q = env.get_q_value(obs, action)
        next_max_q = max([env.get_q_value(new_obs, a) for a in range(env.ACTION_SPACE_VALUES)])
        new_q = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * (reward + DISCOUNT * next_max_q)
        env.update_q_value(obs, action, new_q)

        obs = new_obs
        episode_reward += reward

        if episode % SHOW_EVERY == 0:
            env.render()

    all_episode_rewards.append(episode_reward)

    # 添加这里：收集每个episode的数据
    episode_numbers.append(episode)
    episode_rewards.append(episode_reward)
    episode_steps.append(steps)

    if episode % SHOW_EVERY == 0:
        print(f'Episode #{episode}, epsilon: {epsilon}')
        print(f'Mean reward: {np.mean(all_episode_rewards[-SHOW_EVERY:])}')

    epsilon *= EPS_DECAY

# 保存数据到CSV文件
# import csv
#
# with open('最优——ESBLP_learning_curve_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Episode", "Reward", "Steps"])
#     for ep, rew, steps in zip(episode_numbers, episode_rewards, episode_steps):
#         writer.writerow([ep, rew, steps])
#
# print("Learning curve data saved to '最优——ESBLP_learning_curve_data.csv'")

# 保存数据到CSV文件
import csv
import os

# 确保目标文件夹路径存在
save_path = r'C:\Users\a\Desktop\工作文件夹\文献投稿\嗅觉传感器2.0\实验0826'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 文件完整路径
file_path = os.path.join(save_path, '最优——ESBLP_learning_curve_data.csv')
with open(file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Reward", "Steps"])
    for ep, rew, steps in zip(episode_numbers, episode_rewards, episode_steps):
        writer.writerow([ep, rew, steps])

print(f"Learning curve data saved to '{file_path}'")

# 绘制学习曲线（保持不变）
plt.figure(figsize=(12, 6))
moving_avg = np.convolve(all_episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot(moving_avg)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel(f'Mean {SHOW_EVERY} reward')

# 添加泄漏点检测标记
if env.first_detection_episode:
    plt.axvline(x=env.first_detection_episode, color='r', linestyle='--', label='First Leak Detection')
    plt.text(env.first_detection_episode, plt.ylim()[1], f'First: {env.first_detection_episode}',
             horizontalalignment='right', verticalalignment='top', rotation=90)

if env.second_detection_episode:
    plt.axvline(x=env.second_detection_episode, color='g', linestyle='--', label='Second Leak Detection')
    plt.text(env.second_detection_episode, plt.ylim()[1], f'Second: {env.second_detection_episode}',
             horizontalalignment='right', verticalalignment='top', rotation=90)

plt.legend()
plt.tight_layout()
plt.savefig('最优——ESBLP_learning_curve_plot.png', dpi=300)
plt.show()
# 保存Q_table （保持不变）
with open(f'qtable_{int(time.time())}.pickle', 'wb') as f:
    pickle.dump(env.get_q_table(), f)