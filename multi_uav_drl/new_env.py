from mapM import GridM as MapM
from test import test_data
import os
import copy
from os.path import join as pjoin
import numpy as np
import time

# from pympler import tracker
from gymnasium import spaces

map_x = 16
map_y = 16
max_value = 1.0
min_value = 0.0
channel = 3
num_uavs = 6
init_position = (8, 8)
max_energy = 500
num_action = 2
crange = 1.1
max_distance = 1.0
collection_prop = 0.2
wall_penalty = -1.0
data_reward = 1.0
waste_step = -0.5
alpha = 1.0
epsilon = 1e-4
normalize = 0.1
factor = 0.1


def mypjoin(path1, path2, paths=None):
    full_path = pjoin(path1, path2)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    if paths is not None:
        full_path = pjoin(full_path, paths)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
    return full_path


def myint(a):
    # return int(np.ceil(a))
    return int(np.floor(a))


class Env():
    def __init__(self, log_dir="."):

        # 6-19 00:42
        self.maxaction = 0
        self.minaction = 0
        #

        # self.log_dir = log.full_path
        self.log_dir = log_dir
        # basis
        self.mapx = map_x
        self.mapy = map_y
        self.map = MapM(self.log_dir)
        self.channel = channel
        self.image_data = None
        self.image_position = None

        # uavs
        self.num_uavs = num_uavs
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.map.width, self.map.height, self.channel)) for i in range(self.num_uavs)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(num_action,)) for i in range(self.num_uavs)]
        self.maxenergy = max_energy
        self.crange = crange
        self.maxdistance = max_distance
        self.cspeed = np.float16(collection_prop)
        self.alpha = alpha
        self.track = 1.0 / 1000.0
        self.max_steps = 1000
        self.log_freq = 100
        # ---- 6-8 14:48 add factor
        self.factor = factor
        # ----
        self.epsilon = epsilon
        self.normalize = normalize
        # reward
        self.pwall = wall_penalty
        self.rdata = data_reward
        self.pstep = waste_step

        self.DATAs = np.reshape(test_data, (-1, 3)).astype(np.float16)

        self._mapmatrix = copy.copy(self.DATAs[:, 2])
        self.datas = self.DATAs[:, 0:2] * self.mapx
        self.totaldata = np.sum(self.DATAs[:, 2])
        # log.log(self.DATAs)

        self._image_data = np.zeros((self.map.width, self.map.height)).astype(np.float16)
        self._image_position = np.zeros((num_uavs, self.map.width, self.map.height)).astype(np.float16)
        self.map.draw_wall(self._image_data)
        for i, position in enumerate(self.datas):
            self.map.draw_point(position[0], position[1], self._mapmatrix[i], self._image_data)
        for i_n in range(self.num_uavs):
            # layer 1
            self.map.draw_UAV(init_position[0], init_position[1], 1.0, self._image_position[i_n])
        # self.tr.print_diff()

    def reset(self):
        # initialize data map
        # tr = tracker.SummaryTracker()
        self.mapmatrix = copy.copy(self._mapmatrix)
        # ---- original
        # self.maptrack = np.zeros(self.mapmatrix.shape)
        # ---- new 6-7-11-28
        # self.maptrack = np.ones(self.mapmatrix.shape) * self.track
        # ---- 18:43
        self.maptrack = np.zeros(self.mapmatrix.shape)
        # ----
        # initialize positions of uavs
        self.uav = [list(init_position) for i in range(self.num_uavs)]
        self.eff = [0.0] * self.num_uavs
        self.count = 0
        self.zero = 0

        self.trace = [[] for i in range(self.num_uavs)]
        # initialize remaining energy
        self.energy = np.ones(self.num_uavs).astype(np.float64) * self.maxenergy
        # initialize indicators
        self.collection = np.zeros(self.num_uavs).astype(np.float16)
        self.walls = np.zeros(self.num_uavs).astype(np.int16)

        # time
        self.time_ = 0

        # initialize images
        self.state = self.__init_image()
        # print(self.fairness)
        # image = [np.reshape(np.array([self.image_data, self.image_position[i]]), (self.map.width, self.map.height, self.channel)) for i in range(self.num_uavs)]
        # tr.print_diff()
        return self.__get_state()

    def __init_image(self):
        self.image_data = copy.copy(self._image_data)
        self.image_position = copy.copy(self._image_position)
        # ---- or
        # self.image_track = np.zeros(self.image_position.shape)
        # ---- new 6-7-11-28
        # self.image_track = np.ones(self.image_data.shape) * self.track
        # ---- 18:43
        self.image_track = np.zeros(self.image_position.shape)
        # ----
        state = []
        for i in range(self.num_uavs):
            image = np.zeros((self.map.width, self.map.height, self.channel)).astype(np.float16)
            for width in range(image.shape[0]):
                for height in range(image.shape[1]):
                    image[width][height][0] = self.image_data[width][height]
                    image[width][height][1] = self.image_position[i][width][height]
                    # ---- new 6-7-11-28
                    # image[width][height][2] = self.image_track[width][height]
                    # ---- end new
            state.append(image)
        return state

    def __draw_image(self, clear_uav, update_point, update_track):
        for n in range(self.num_uavs):
            for i, value in update_point:
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 0])
            self.map.clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[n][:, :, 1])
            self.map.draw_UAV(self.uav[n][0], self.uav[n][1], self.energy[n] / self.maxenergy, self.state[n][:, :, 1])
            # ---- draw track
            for i, value in update_track:
                # ---- or
                self.map.draw_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 2])
                # ---- new 6-7 16:15
                # self.map.draw_point(self.datas[i][0], self.datas[i][1], -1. * value, self.state[n][:,:,2])
                # ---- end new

    def __get_state(self):
        return copy.deepcopy(self.state)

    def __get_reward(self, value, distance, fairness, fairness_):
        if value != 0:
            # ---- or
            # factor0 = value / (distance + self.alpha * value + self.epsilon)
            # ---- 6-8 14:48
            factor0 = value / (self.factor * distance + self.alpha * value + self.epsilon)
            # ----
            # delta_fairness = np.fabs(fairness_ - fairness)
            # print(delta_fairness)
            return factor0 * fairness_
        else:
            return -1.0 * self.normalize * distance

    def __get_fairness(self, values):
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.0
        jain_fairness_index = square_of_sum / sum_of_square / float(len(values))
        return jain_fairness_index

    def __get_eff(self, value, distance):
        return value / self.maxenergy

    def __get_eff1(self, value, distance):
        return value / (distance + self.alpha * value + self.epsilon)

    def __cusume_energy0(self, uav, value, distance):
        self.energy[uav] -= distance

    def __cusume_energy1(self, uav, value, distance, energy=None):
        if energy is None:
            # ---- or
            # self.erengy[uav] -= (distance + value * self.alpha)
            # ---- 6-8 14:48
            self.energy[uav] -= self.factor * distance + self.alpha * value
            # ----
        else:
            # ---- or
            # energy[uav] -= (distance + value * self.alpha)
            # ---- 14:48
            energy[uav] -= self.factor * distance + self.alpha * value
            # ----

    # ---- or
    # def step(self, action):
    #     self.count += 1
    #     actions = copy.deepcopy(action)
    #     normalize = self.normalize
    #     for i in range(self.n):
    #         for ii in actions[i]:
    #             if np.isnan(ii):
    #                 print('Nan')
    #                 while True:
    #                     pass
    # ---- 6-8 10:57
    def step(self, actions, indicator=None):
        self.count += 1
        action = copy.deepcopy(actions)
        # 6-20 00:43
        if np.max(action) > self.maxaction:
            self.maxaction = np.max(action)
            # print(self.maxaction)
        if np.min(action) < self.minaction:
            self.minaction = np.min(action)
            # print(self.minaction)
        action = np.clip(action, -1e3, 1e3)  #

        normalize = self.normalize

        # Check for NaN values
        for i in range(self.num_uavs):
            for ii in action[i]:
                if np.isnan(ii):
                    raise ValueError("NaN value detected in action")

        reward = [0] * self.num_uavs
        self.dn = [False] * self.num_uavs  # no energy UAV
        update_points = []
        update_tracks = []
        clear_uav = copy.copy(self.uav)
        new_positions = []
        c_f = self.__get_fairness(self.maptrack)
        # update positions of UAVs
        for i in range(self.num_uavs):
            self.trace[i].append(self.uav[i])
            distance = np.sqrt(np.power(action[i][0], 2) + np.power(action[i][1], 2))
            data = 0

            if distance <= self.maxdistance and self.energy[i] >= distance:
                new_x = self.uav[i][0] + action[i][0]
                new_y = self.uav[i][1] + action[i][1]
            else:
                maxdistance = self.maxdistance if self.maxdistance <= self.energy[i] else self.energy[i]
                if distance <= self.epsilon:
                    distance = self.epsilon
                    print("very small.")
                new_x = self.uav[i][0] + maxdistance * action[i][0] / distance
                new_y = self.uav[i][1] + maxdistance * action[i][1] / distance
                distance = maxdistance
            if distance <= self.epsilon:
                self.zero += 1
            self.__cusume_energy1(i, 0, distance)
            if 0 <= new_x < self.mapx and 0 <= new_y < self.mapy:
                new_positions.append([new_x, new_y])
            else:
                new_positions.append([self.uav[i][0], self.uav[i][1]])
                reward[i] += normalize * self.pwall
                self.walls[i] += 1
            # calculate distances between UAV and data points
            _pos = np.repeat([new_positions[-1]], [self.datas.shape[0]], axis=0)
            _minus = self.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                if np.sqrt(dis) <= self.crange:
                    self.maptrack[index] += self.track
                    update_tracks.append([index, self.maptrack[index]])
                    if self.mapmatrix[index] > 0:
                        data += self._mapmatrix[index] * self.cspeed
                        self.mapmatrix[index] -= self._mapmatrix[index] * self.cspeed
                        if self.mapmatrix[index] < 0:
                            self.mapmatrix[index] = 0.0
                        update_points.append([index, self.mapmatrix[index]])
            # update info
            value = data if self.energy[i] >= data * self.alpha else self.energy[i]
            self.__cusume_energy1(i, value, 0.0)
            c_f_ = self.__get_fairness(self.maptrack)
            # ---- 6-7
            # 11:32
            # reward[i] += self.__get_reward9(value, distance, c_f, c_f_)
            # ---- 11:44
            # reward[i] += self.__get_reward7(value, distance, c_f, c_f_)
            # ---- 18:43
            reward[i] += self.__get_reward(value, distance, c_f, c_f_)
            # ----
            c_f = c_f_
            self.eff[i] += self.__get_eff1(value, distance)
            self.collection[i] += value
            if self.energy[i] <= self.epsilon * self.maxenergy:
                self.dn[i] = True
        self.uav = new_positions
        t = time.time()
        self.__draw_image(clear_uav, update_points, update_tracks)
        self.time_ += time.time() - t
        # ---- or
        reward = list(np.clip(np.array(reward) / normalize, -2.0, 1.0))
        # ---- new 18:43
        # reward = list(np.clip(np.array(reward) / normalize, -1., 1.))
        # ---- end new
        info = None
        state = self.__get_state()
        # Check for NaN rewards
        for r in reward:
            if np.isnan(r):
                raise ValueError("NaN value detected in reward")
            
        print(f"state's shape = {state.shape}")

        return state, reward, sum(self.dn), info, indicator

    def render(self):
        print("coding...")

    @property
    def leftrewards(self):
        return np.sum(self.mapmatrix) / self.totaldata

    @property
    def efficiency(self):
        return np.sum(self.collection / self.totaldata) / (self.num_uavs - np.sum(self.normal_energy)) * self.collection_fairness

    @property
    def normal_energy(self):
        return list(np.array(self.energy) / self.maxenergy)

    @property
    def fairness(self):
        square_of_sum = np.square(np.sum(self.mapmatrix[:]))
        sum_of_square = np.sum(np.square(self.mapmatrix[:]))
        fairness = square_of_sum / sum_of_square / float(len(self.mapmatrix))
        return fairness

    @property
    def collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness

    @property
    def normal_collection_fairness(self):
        collection = self._mapmatrix - self.mapmatrix
        for index, i in enumerate(collection):
            collection[index] = i / self._mapmatrix[index]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness
