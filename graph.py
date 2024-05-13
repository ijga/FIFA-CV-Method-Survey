import math
import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def euclidean_distance(player1, player2):
    x1, y1 = player1.x, player1.y
    x2, y2 = player2.x, player2.y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def angle_to(x1, y1, x2, y2):
    # Calculate the differences in coordinates
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the angle using arctan
    angle_rad = math.atan2(dy, dx)
    
    # Convert angle from radians to degrees
    angle_deg = math.degrees(angle_rad)
    
    # Ensure the angle is positive (between 0 and 360 degrees)
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

class Player:
    def __init__(self, idx, team, x, y):
        self.id = int(idx)
        self.team = team

        self.x = float(x)
        self.y = float(y)

    def __str__(self) -> str:
        return f'(Player {self.id}: {self.team}; ({self.x}, {self.y}))'

class Goal:
    def __init__(self, idx, side, x, y) -> None:
        self.id = idx
        self.side = side
        self.x = x
        self.y = y

    def create_bounds():
        pass

    def __str__(self) -> str:
        return f'({self.side} Goal ({self.x}, {self.y}))'

class Ball:
    def __init__(self, idx, x, y) -> None:
        self.id = idx
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'(Ball: ({self.x}, {self.y}))'

class Edge:
    def __init__(self, idx, angle, type, player, target):
        self.id = int(idx)
        self.angle = angle
        self.type = type # team to team, team to other team, team to goal, team to ball
        self.player = player # player
        self.target = target
    
    def create_feature_vector(self):
        return [self.angle, self.target, self.type, self]

    def __str__(self) -> str:
        return f'(Edge: {self.id} {self.angle} {self.type}; {self.player} -> {self.target})'

class Graph:
    def __init__(self):
        self.man_city = {}
        self.man_city_gk = []
        self.man_utd = {}
        self.man_utd_gk = []
        self.goal = []
        self.ball = []
        # self.vertices = []
        self.edges = defaultdict(list)

    def clear(self):
        self.man_city = {}
        self.man_city_gk = []
        self.man_utd = {}
        self.man_utd_gk = []
        self.goal = []
        self.ball = []
        # self.vertices = []
        self.edges = defaultdict(list)

    def add_player(self, player):
        if player.team == 'man_city':
            
            self.man_city[player.id] = player
        elif player.team == 'man_utd':
            self.man_utd[player.id] = player

    def add_gk(self, player):
        if player.team[:-3] == 'man_city':
            self.man_city_gk = [player]
        elif player.team[:-3] == 'man_utd':
            self.man_utd_gk = [player]

    def add_goal(self, goal):
        if not self.goal:
            self.goal = [goal]
        else:
            if goal.side == self.goal[0].side:
                if goal.side == "right":
                    pass
                elif goal.side == "left":
                    pass

    def add_ball(self, ball):
        self.ball = [ball]

    def add_edges(self, degree):
        idx = 0
        # directed graph
        for player in self.man_city.values():
            for target in self.man_city.values():
                idx += 1
                if player.id != target.id:
                    angle = angle_to(player.x, player.y, target.x, target.y)
                    self.edges[player.id].append(Edge(idx, angle, 'team', player, target))

        for player in self.man_city.values():
            distances = []

            for target in self.man_utd.values():
                distance = euclidean_distance(player, target)
                heapq.heappush(distances, (distance, target.id))

            for i in range(degree):
                idx += 1
                closest_distance, target = heapq.heappop(distances)
                angle = angle_to(player.x, player.y, self.man_utd[target].x, self.man_utd[target].y)
                self.edges[player.id].append(Edge(idx, angle, 'opponent', player, self.man_utd[target]))

        if self.ball:
            distances = []

            ball = self.ball[0]
                # find closest players to goal
            for player in self.man_city.values():
                distance = euclidean_distance(player, ball)
                heapq.heappush(distances, (distance, ball))

            idx += 1
            closest_distance, ball = heapq.heappop(distances)
            angle = angle_to(player.x, player.y, ball.x, ball.y)
            self.edges[player.id].append(Edge(idx, angle, 'ball', player, ball))

        if self.man_utd_gk:
            if self.goal:
                goal = self.goal[0]
                # find closest players to goal
                for player in self.man_city:
                    distance = euclidean_distance(player, goal)
                    heapq.heappush(distances, (distance, goal))
                
                for i in range(degree):
                    idx += 1
                    closest_distance, goal = heapq.heappop(distances)
                    angle = angle_to(player.x, player.y, goal.x, goal.y)
                    self.edges[player.id].append(Edge(idx, angle, 'goal', player, goal))

    def visualize_graph(self):
        G = nx.Graph()

        city = {player.id: (player.x, -1 * player.y) for player in self.man_city.values()}
        utd = {player.id: (player.x, -1 * player.y) for player in self.man_utd.values()}
        ball = {ball.id: (ball.x, -1 * ball.y) for ball in self.ball}
        game_objects = {**city, **utd, **ball}
        if self.man_city_gk:
            gk = self.man_city_gk[0]
            game_objects.update({gk.id: (gk.x, -1 * gk.y)})
        elif self.man_utd_gk:
            gk = self.man_utd_gk[0]
            game_objects.update({gk.id: (gk.x, -1 * gk.y)})

        for vertex in game_objects:
            G.add_node(vertex)

        for src, edges in self.edges.items():
            for edge in edges:
                G.add_edge(src, edge.target.id)

        fig, ax = plt.subplots()
        pos = game_objects
        nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
        plt.show()

    def __str__(self) -> str:
        return "game_state"
