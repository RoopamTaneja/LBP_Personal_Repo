from PIL import Image
import time
import os
import numpy as np

image_size = 80
wall_value = -1
wall_width = 4
map_x = 16
map_y = 16


class Grid(object):
    def __init__(self, width, height):
        # self.__map = np.zeros((width, height))
        self.width = width
        self.height = height

    def draw_sqr(self, x, y, width, height, value, map):
        assert 0 <= x < self.width and 0 <= y < self.height, f"Invalid position: ({x}, {y})"
        for i in range(x, x + width, 1):
            for j in range(y, y + height, 1):
                map[i][j] = value

    # def get_value(self, x, y, map):
    # return map[x][y]


class GridM(Grid):

    def __init__(self, log_path, width=80, height=80):
        super(GridM, self).__init__(width, height)
        self.__time = time.time()
        self.full_path = os.path.join(log_path, "img_map")
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

    def draw_wall(self, map):
        wall = wall_value
        width = wall_width
        for j in range(0, 80, 1):
            for i in range(80 - width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for i in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
        for i in range(0, 80, 1):
            for j in range(0, width, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)
            for j in range(80 - width, 80, 1):
                self.draw_sqr(i, j, 1, 1, wall, map)

    # def get_value(self, x, y, map):
    # x, y = self.__trans(x, y)
    # return super( GridM, self).get_value(x, y, map)

    def __trans(self, x, y):
        return int(4 * x + wall_width * 2), int(y * 4 + wall_width * 2)

    def draw_obstacle(self, x, y, width, height, map):
        # self.clear_cell(x, y, map)
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, width * 4, height * 4, wall_value, map)


    # xy transpose occur
    def draw_point(self, x, y, value, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, value, map)

    def clear_point(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 2, 2, 0, map)

    def clear_uav(self, x, y, map):
        x, y = self.__trans(x, y)
        self.draw_sqr(x, y, 4, 4, 0, map)

    def draw_UAV(self, x, y, value, map):
        x = -1 if x < -1 else map_x if x > map_x else x
        y = -1 if y < -1 else map_y if y > map_y else y
        self.draw_sqr(x, y, 4, 4, value, map)

    def save_as_png(self, map, ip=None):
        img = Image.fromarray(map * 255)
        img = img.convert("L")
        img.show()
        if ip is None:
            name = time.time() - self.__time
        else:
            name = str(ip)
        img.save(os.path.join(self.full_path, f"{name}.png"), "png")


def driver_function():
    # Create a  GridM instance
    new_img = GridM("log_directory")

    test_map = np.zeros((80, 80))

    # Draw the walls
    new_img.draw_wall(test_map)


    # Draw some UAVs
    new_img.draw_UAV(12, 8, 0.7, test_map)
    new_img.draw_UAV(12, 10, 0.7, test_map)


    # Draw some points
    new_img.draw_point(7, 7, 1, test_map)
    new_img.draw_point(13, 13, 1, test_map)

    # Save the map as a PNG
    new_img.save_as_png(test_map, "example_map")

    print(" Grid saved successfully!")


# Run the driver function
if __name__ == "__main__":
    driver_function()
