import math
import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

file_path = "points/points.txt"

class Player:
    def __init__(self, idx, team, x, y):
        self.id = int(idx)
        self.team = float(team)
        self.x = float(x)
        self.y = float(y)

    def __str__(self) -> str:
        return f'{self.team}: ({self.x}, {self.y})'

def euclidean_distance(player1, player2):
    x1, y1 = player1.x, player1.y
    x2, y2 = player2.x, player2.y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = defaultdict(list)

    def add_point(self, player):
        self.vertices.append(player)

    def add_edges(self, degree):
        # find distance to x closest, undirected graph
        
        for player in self.vertices:
            distances = []
            for other_player in self.vertices:
                if player.id == other_player.id:
                    continue
                distance = euclidean_distance(player, other_player)
                heapq.heappush(distances, (distance, other_player.id))
        
            for i in range(degree):
                closest_distance, id = heapq.heappop(distances)
                if id not in self.edges[player.id]:
                    self.edges[player.id].append(id)
                if player.id not in self.edges[id]:
                    self.edges[id].append(player.id)    
                print(closest_distance, id)

        print(self.edges)

    def visualize_graph(self):
        G = nx.Graph()

        pos = {player.id:(player.x, -1*player.y) for player in self.vertices}

        for vertex in self.vertices:
            G.add_node(vertex.id)

        for src, array in self.edges.items():
            for dest in array:
                G.add_edge(src, dest)

        # Draw the graph
        nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
        plt.show()

    def __str__(self) -> str:
        return "game_state"


game_state = Graph()
# Open the file for reading
with open(file_path, "r") as file:
    # Read the contents of the file
    file_contents = file.read()
    file_lines = file_contents.split("\n")
    for idx, line in enumerate(file_lines):
        classification, point = line.split(" + ")
        x, y = point[1:-2].split(',') # from top left as 0, 0
        player = Player(idx, classification, x, y)
        game_state.add_point(player)
        print(player)

    game_state.add_edges(3)
    game_state.visualize_graph()
