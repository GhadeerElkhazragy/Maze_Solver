# Packages
import cv2
import numpy as np
import threading
import colorsys
import sys
from PIL import Image

# Point class used in BFS


class Point(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


rw = 2
p = 0
start = Point()
end = Point()

# BFS neighbors
dir4 = [Point(0, -1), Point(0, 1), Point(1, 0), Point(-1, 0)]


# BFS algorithm
def BFS(s, e):
    global img, h, w
    const = 10000  # trial and error

    found = False
    queue = []
    visitor = [[0 for j in range(w)] for i in range(h)]
    parent = [[Point() for j in range(w)] for i in range(h)]

    queue.append(s)
    visitor[s.y][s.x] = 1
    while len(queue) > 0:
        p = queue.pop(0)
        for d in dir4:
            node = p + d
            if (not_blocked(node) and visitor[node.y][node.x] == 0):
                queue.append(node)
                visitor[node.y][node.x] = visitor[p.y][p.x] + 1

                img[node.y][node.x] = list(reversed(
                    [i * 255 for i in colorsys.hsv_to_rgb(visitor[node.y][node.x] / const, 1, 1)])
                )
                parent[node.y][node.x] = p
                if node == e:
                    found = True
                    del queue[:]
                    break

    path = []
    if found:
        p = e
        while p != s:
            path.append(p)
            p = parent[p.y][p.x]
        path.append(p)
        path.reverse()

        for p in path:
            img[p.y][p.x] = [255, 255, 255]
            img[p.y + 1][p.x + 1] = [255, 255, 255]
            img[p.y + 2][p.x + 2] = [255, 255, 255]

        print("Path Found")

    else:
        print("Path Not Found")

# getting start and end points


def mouse_event(event, pX, pY, flags, param):
    global img, start, end, p, s, e

    if event == cv2.EVENT_LBUTTONUP:
        if p == 0:
            cv2.rectangle(img, (pX - rw, pY - rw),
                          (pX + rw, pY + rw), (0, 0, 255), -1)
            start = Point(pX, pY)
            s = (start.x, start.y)
            print("start = ", start.x, start.y)
            p += 1
        elif p == 1:
            cv2.rectangle(img, (pX - rw, pY - rw),
                          (pX + rw, pY + rw), (0, 200, 50), -1)
            end = Point(pX, pY)
            e = (end.x, end.y)
            print("end = ", end.x, end.y)
            p += 1


def disp():
    global img
    cv2.imshow("Image", img)
    cv2.setMouseCallback('Image', mouse_event)
    while True:
        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Detecting maze walls bfs


def not_blocked(node):
    if (node.x >= 0 and node.x < w and node.y >= 0 and node.y < h and
            (img[node.y][node.x][0] != 0 or img[node.y][node.x][1] != 0 or img[node.y][node.x][2] != 0)):
        return True


# A Star neighbors


def von_neumann_neighbors(p):
    x, y = p
    neighbors = [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]
    return [p for p in neighbors if not_blocked_a(p)]

# cost calculation


def squared_euclidean(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# Main
def solve_bfs(image):
    global img,h,w
    img = image
    # img = cv2.imread("maze1.jpg", cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    print("Select start and end points : ")



    t = threading.Thread(target=disp, args=())
    t.daemon = True
    t.start()

    while p < 2:
        pass

    BFS(start, end)
    # A_STAR(s, e, von_neumann_neighbors, distance, heuristic)


    cv2.waitKey(0)




