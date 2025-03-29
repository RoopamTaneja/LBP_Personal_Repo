from data_points import test_data
import copy
import numpy as np
from gymnasium import spaces
from PIL import Image
import os

# Environment parameters
map_width = map_height = 16
grid_width = grid_height = 80
wall_width = 4
wall_value = -1
channel = 3
num_uavs = 6
init_positions = [[4, 4], [12, 4], [4, 12], [12, 12], [8, 8], [8, 4]]
max_energy = 500
num_action = 2
comm_range = 1.1
max_distance = 1.0
collection_prop = 0.2
wall_penalty = -1.0
data_reward = 1.0
waste_step_penalty = -0.5
alpha = 1.0
epsilon = 1e-4
normalize = 0.1
factor = 0.1


class Env:
    def __init__(self, log_dir="."):
        self.map_width = map_width
        self.map_height = map_height
        self.width = grid_width
        self.height = grid_height
        self.channel = channel
        self.img_path = log_dir
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        # UAV configuration
        self.num_uavs = num_uavs
        self.observation_space = [spaces.Box(low=-1, high=1, shape=(self.width, self.height, self.channel)) for _ in range(self.num_uavs)]
        self.action_space = [spaces.Box(low=-1, high=1, shape=(num_action,)) for _ in range(self.num_uavs)]

        # Movement and collection parameters
        self.maxenergy = max_energy
        self.comm_range = comm_range
        self.maxdistance = max_distance
        self.cspeed = np.float16(collection_prop)
        self.alpha = alpha
        self.track = 1.0 / 1000.0
        self.factor = factor
        self.epsilon = epsilon
        self.normalize = normalize
        self.max_steps = 1000
        self.log_freq = 100
        self.dn = [False] * self.num_uavs  # UAVs with depleted energy

        # Reward parameters
        self.pwall = wall_penalty
        self.rdata = data_reward
        self.pstep = waste_step_penalty

        # Initialize data points from test_data module
        self.DATAs = np.reshape(test_data, (-1, 3)).astype(np.float16)
        self._mapmatrix = copy.copy(self.DATAs[:, 2])
        self.totaldata = np.sum(self._mapmatrix)
        self.datas = self.DATAs[:, 0:2] * map_width

        self._init_data_map = np.zeros((self.width, self.height)).astype(np.float16)
        self._init_position_map = np.zeros((num_uavs, self.width, self.height)).astype(np.float16)

        # Draw walls and data points on data map
        self._draw_wall(self._init_data_map)
        for i, position in enumerate(self.datas):
            self._draw_data_point(position[0], position[1], self._mapmatrix[i], self._init_data_map)

        # Draw initial UAV positions
        for i_n in range(self.num_uavs):
            self._draw_UAV(init_positions[i_n][0], init_positions[i_n][1], 1.0, self._init_position_map[i_n])

    def _transform_coords(self, x, y):
        """Transform logical coordinates to visual coordinates"""
        return int(4 * x + wall_width * 2), int(4 * y + wall_width * 2)

    def _draw_square(self, x, y, width, height, value, grid, add = False):
        for i in range(x, x + width):
            for j in range(y, y + height):
                if 0 <= i < self.width and 0 <= j < self.height:
                    if add:
                        grid[i][j] += value
                    else:
                        grid[i][j] = value

    def _draw_wall(self, grid):
        for j in range(self.height):
            for i in range(wall_width):
                grid[i][j] = wall_value
            for i in range(self.height - wall_width, self.height):
                grid[i][j] = wall_value
        for i in range(self.width):
            for j in range(wall_width):
                grid[i][j] = wall_value
            for j in range(self.height - wall_width, self.height):
                grid[i][j] = wall_value

    def _draw_data_point(self, x, y, value, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 2, 2, value, grid, add=True)

    def _draw_UAV(self, x, y, value, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, value, grid)

    def _clear_data_point(self, x, y, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 2, 2, 0, grid)

    def _clear_uav(self, x, y, grid):
        x, y = self._transform_coords(x, y)
        self._draw_square(x, y, 4, 4, 0, grid)

    def save_image(self, name=None, include_uavs=True):
        grid = self.image_data.copy()
        if np.min(grid) < 0:
            # Map from [-1,1] to [0,1]
            grid = (grid + 1) / 2
        rgb_img = np.stack([grid, grid, grid], axis=2)
        if include_uavs:
            colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
            for i, pos in enumerate(self.uav_pos):
                x, y = self._transform_coords(pos[0], pos[1])
                color = colors[i % len(colors)]
                for dx in range(4):
                    for dy in range(4):
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                            rgb_img[x + dx, y + dy] = color
        img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img, "RGB")
        if name is None:
            name = "initial_state"
        img.save(os.path.join(self.img_path, f"{name}.png"), "png")

    def set_initial_state(self, initial_state):
        """
        Set the initial state of the environment from an image-processed grid representation
        """
        # Validate input shape
        if initial_state.shape[0] != self.width or initial_state.shape[1] != self.height or initial_state.shape[2] != self.channel:
            raise ValueError(f"Expected state shape ({self.width}, {self.height}, {self.channel}), " f"got {initial_state.shape}")

        # Extract obstacle/wall information from channel 0
        self._image_data = initial_state[:, :, 0].astype(np.float16)

        # Process DATAs based on the data points in the image
        # Find data points in the image (based on threshold)
        data_points = []
        data_values = []

        # Extract data points from the image (channel 2)
        data_mask = initial_state[:, :, 2] > 0.5  # Threshold for data points

        y_indices, x_indices = np.where(data_mask)
        for i in range(len(y_indices)):
            x, y = x_indices[i], y_indices[i]
            data_points.append([x, y])
            data_values.append(initial_state[y, x, 2])

        # Update internal data structures
        self.datas = np.array(data_points).astype(np.float16)
        self._mapmatrix = np.array(data_values).astype(np.float16)
        self.totaldata = np.sum(self._mapmatrix)

        # Update DATAs for consistency
        self.DATAs = np.zeros((len(data_points), 3)).astype(np.float16)
        self.DATAs[:, 0:2] = self.datas / self.map_width  # Normalize coordinates
        self.DATAs[:, 2] = self._mapmatrix

        # Extract UAV positions from channel 1
        uav_positions = []

        # Find UAV positions in the image (based on threshold)
        uav_mask = initial_state[:, :, 1] > 0.5  # Threshold for UAV positions

        y_indices, x_indices = np.where(uav_mask)
        for i in range(min(len(y_indices), self.num_uavs)):
            x, y = x_indices[i], y_indices[i]
            uav_positions.append([x, y])

        # If we found fewer UAVs than expected, add default positions
        while len(uav_positions) < self.num_uavs:
            idx = len(uav_positions) % len(init_positions)
            uav_positions.append(list(init_positions[idx]))

        # Update initial UAV positions
        self._init_position_map = np.zeros((self.num_uavs, self.width, self.height)).astype(np.float16)

        for i, pos in enumerate(uav_positions):
            if i < self.num_uavs:
                self._draw_UAV(pos[0], pos[1], 1.0, self._init_position_map[i])

        # Rebuild the initial state images
        self.__init_state()
        print(f"Environment initialized from image: {len(data_points)} data points, {len(uav_positions)} UAV positions")
        return copy.deepcopy(self.state)

    def __init_state(self):
        """Initialize image representation for each UAV"""
        self.image_data = copy.copy(self._init_data_map)
        self.image_position = copy.copy(self._init_position_map)
        self.image_track = np.zeros(self.image_position.shape)

        # Create state representation for each UAV
        state = []
        for i in range(self.num_uavs):
            image = np.zeros((self.width, self.height, self.channel)).astype(np.float16)
            image[:, :, 0] = self.image_data
            image[:, :, 1] = self.image_position[i]
            # image[:, :, 2] is already initialized to zeros
            state.append(image)
        self.state = state

    def __update_state(self, clear_uav, update_point, update_track):
        """Update state representation after UAV movements and data collection"""
        for n in range(self.num_uavs):
            # Update data points (channel 0)
            for i, value in update_point:
                self._clear_data_point(self.datas[i][0], self.datas[i][1], self.state[n][:, :, 0])
            for i, value in update_point:
                self._draw_data_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 0])

            # Update UAV positions (channel 1)
            self._clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[n][:, :, 1])
            self._draw_UAV(self.uav_pos[n][0], self.uav_pos[n][1], self.energy[n] / self.maxenergy, self.state[n][:, :, 1])

            # Update track information (channel 2)
            for i, value in update_track:
                self._clear_data_point(self.datas[i][0], self.datas[i][1], self.state[n][:, :, 2])
            for i, value in update_track:
                self._draw_data_point(self.datas[i][0], self.datas[i][1], value, self.state[n][:, :, 2])

    def __get_reward(self, value, distance, fairness, fairness_):
        """Calculate reward based on data collected, distance moved, and fairness"""
        if value != 0:  # If data was collected
            factor0 = value / (self.factor * distance + self.alpha * value + self.epsilon)
            return factor0 * fairness_
        else:  # If no data was collected
            return -1.0 * self.normalize * distance

    def __get_fairness(self, values):
        """Calculate Jain's fairness index for a set of values"""
        square_of_sum = np.square(np.sum(values))
        sum_of_square = np.sum(np.square(values))
        if sum_of_square == 0:
            return 0.0
        jain_fairness_index = square_of_sum / sum_of_square / float(len(values))
        return jain_fairness_index

    def __get_efficiency(self, value, distance):
        """Calculate efficiency of data collection"""
        return value / (distance + self.alpha * value + self.epsilon)

    @property
    def leftrewards(self):
        """Proportion of data remaining to be collected"""
        return np.sum(self.mapmatrix) / self.totaldata

    @property
    def efficiency(self):
        """Overall efficiency of the data collection process"""
        active_uavs = self.num_uavs - np.sum(self.normal_energy)
        if active_uavs == 0:
            return 0
        return np.sum(self.collection / self.totaldata) / active_uavs * self.collection_fairness

    @property
    def normal_energy(self):
        """Normalized energy levels for all UAVs"""
        return list(np.array(self.energy) / self.maxenergy)

    @property
    def fairness(self):
        """Fairness of remaining data distribution"""
        square_of_sum = np.square(np.sum(self.mapmatrix[:]))
        sum_of_square = np.sum(np.square(self.mapmatrix[:]))
        if sum_of_square == 0:
            return 1.0
        return square_of_sum / sum_of_square / float(len(self.mapmatrix))

    @property
    def collection_fairness(self):
        """Normalized fairness of data collection (proportional to original values)"""
        collection = self._mapmatrix - self.mapmatrix
        # Normalize each collected value by the original data point value
        for index, i in enumerate(collection):
            if self._mapmatrix[index] > 0:
                collection[index] = i / self._mapmatrix[index]
            else:
                collection[index] = 0
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        if sum_of_square == 0:
            return 0.0
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness

    def step(self, actions):
        """Process one step of the environment given agent actions"""
        action = copy.deepcopy(actions)

        # Check for invalid actions
        for i in range(self.num_uavs):
            if np.isnan(action[i]).any():
                raise ValueError("NaN value detected in action")

        # Initialize step variables
        reward = [0] * self.num_uavs
        update_points = []
        update_tracks = []
        clear_uav = copy.copy(self.uav_pos)
        new_positions = []

        # Calculate initial fairness
        c_f = self.__get_fairness(self.maptrack)

        # Process each UAV's action
        for i in range(self.num_uavs):
            # Record trajectory
            if self.dn[i]:
                new_positions.append(self.uav_pos[i])
                continue

            # Action[0] is angle in radians (scaled from [-1,1] to [0,2π])
            # Action[1] is distance ratio (scaled from [-1,1] to [0,1])
            angle = (action[i][0] + 1) * np.pi  # Map from [-1,1] to [0,2π]
            distance_ratio = (action[i][1] + 1) / 2  # Map from [-1,1] to [0,1]

            distance = distance_ratio * self.maxdistance
            # Limit movement based on available energy
            if self.energy[i] < distance:
                distance = distance_ratio * self.energy[i]
            delta_x = int(distance * np.cos(angle))
            delta_y = int(distance * np.sin(angle))
            data = 0

            new_x = self.uav_pos[i][0] + delta_x
            new_y = self.uav_pos[i][1] + delta_y

            # Check boundary constraints
            if 0 <= new_x < self.map_width and 0 <= new_y < self.map_height:
                new_positions.append([new_x, new_y])
            else:
                # Stay in place and apply wall penalty
                new_positions.append([self.uav_pos[i][0], self.uav_pos[i][1]])
                reward[i] += self.normalize * self.pwall
                self.wall_collisions[i] += 1

            # Calculate distances to all data points
            _pos = np.repeat([new_positions[-1]], [self.datas.shape[0]], axis=0)
            _minus = self.datas - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)

            # Process data collection for points within range
            for index, dis in enumerate(_dis):
                if np.sqrt(dis) <= self.comm_range:
                    # Update track for visited points
                    self.maptrack[index] += self.track
                    update_tracks.append([index, self.maptrack[index]])

                    # Collect data if available
                    if self.mapmatrix[index] > 0:
                        data += self._mapmatrix[index] * self.cspeed
                        self.mapmatrix[index] -= self._mapmatrix[index] * self.cspeed
                        if self.mapmatrix[index] < 0:
                            self.mapmatrix[index] = 0.0
                        update_points.append([index, self.mapmatrix[index]])

            # Limit data collection by available energy
            value = min(data, self.energy[i] / self.alpha) if data > 0 else 0

            # Consume energy for movement and data collection
            self.energy[i] -= self.factor * distance + self.alpha * value

            # Calculate new fairness after this UAV's action
            c_f_ = self.__get_fairness(self.maptrack)

            # Calculate reward
            reward[i] += self.__get_reward(value, distance, c_f, c_f_)
            c_f = c_f_

            # Update efficiency and collection metrics
            self.eff[i] += self.__get_efficiency(value, distance)
            self.collection[i] += value

            # Check if energy is depleted
            if self.energy[i] <= self.epsilon * self.maxenergy:
                self.dn[i] = True

        # Update UAV positions
        self.uav_pos = new_positions

        # Update state representation
        self.__update_state(clear_uav, update_points, update_tracks)

        # Check for invalid rewards
        for r in reward:
            if np.isnan(r):
                raise ValueError("NaN value detected in reward")

        # TO CHECK
        # Calculate metrics for return
        coverage = 1.0 - self.leftrewards
        fairness = self.__get_fairness(self.maptrack)
        energy_efficiency = np.mean(self.eff)

        # Calculate communication penalties
        comm_penalties = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                dist = np.sqrt(np.sum(np.power(np.array(self.uav_pos[i]) - np.array(self.uav_pos[j]), 2)))
                if dist > self.comm_range:
                    comm_penalties[i] += 0.1
                    comm_penalties[j] += 0.1

        done = sum(self.dn) == num_uavs  # Done if all UAVs are depleted

        # Return state, done flag, reward, and metrics
        return (copy.deepcopy(self.state), done, reward, (coverage, fairness, energy_efficiency, comm_penalties))

    def reset(self):
        """Reset environment to initial state for a new episode"""
        # Reset data matrix and tracking
        self.mapmatrix = copy.copy(self._mapmatrix)
        self.maptrack = np.zeros(self.mapmatrix.shape)

        # Reset UAV positions and stats
        self.uav_pos = copy.deepcopy(init_positions)
        self.eff = [0.0] * self.num_uavs

        # Reset energy and performance indicators
        self.energy = np.ones(self.num_uavs).astype(np.float64) * self.maxenergy
        self.collection = np.zeros(self.num_uavs).astype(np.float16)
        self.wall_collisions = np.zeros(self.num_uavs).astype(np.int16)
        self.dn = [False] * self.num_uavs 

        # Initialize images and state representation
        self.__init_state()
        return copy.deepcopy(self.state)


if __name__ == "__main__":
    env = Env()
    env.reset()
    # env.save_image()