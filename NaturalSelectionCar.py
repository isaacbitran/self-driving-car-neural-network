import pygame 
from numpy import exp, array, random, dot, tanh
# import numpy as np
from copy import deepcopy
import math
import matplotlib.pyplot as plt

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

track_color = (0, 0, 0)

WIDTH, HEIGHT = 600, 600

ipopulation = 300
population = 50
total_steps = 1000 #should be 1000
acceleration = 0.4
rotSpeed = 0.1
tick = 60

record = []

mutation = 0.1 #0.05?
mutation_frequency = 0.05
speciation_rate = 1/1000

initVel = 4
max_speed = 15
min_speed = 2

initX = 0
initY = 0
initAngle = 0

max_sensor = 130

drawLines = False

INPUTS = 6
H1 = 4
# H2 = 4
OUTPUTS = 2 #Turn Left, turn right
    
class Car:
    def __init__(self):
        self.reset()
        # random.seed(1)
        self.weights1 = 2 * random.random((INPUTS, H1)) - 1
        self.weights2 = 2 * random.random((H1, OUTPUTS)) - 1

        self.biases1 = 2 * random.random((1, H1)) - 1
        self.biases2 = 2 * random.random((1, OUTPUTS)) - 1

    #Saved most optimal best car
#         w = [[[0.9824248069035761, 0.4221479008106053, 1.9665282149558667, 1.3812854148850693], [-0.8871905616242814, -1.468320327587861, 0.4082750066994909, -1.309995706711701], [0.7044319772584723, 0.3517352734700928, 0.14300020414916786, 0.551000650694441], [-1.4961835063221087, -1.649096437047167, -0.3090687794418283, 1.1808210526355172], [1.940649088111735, 1.024999505097848, 0.3726989336895623, -0.3542683039669771], [-1.387987949941524, -0.28612429273578727, -3.2560363606152167, -1.769595975905002]],
# [[-3.027012254647096, 1.173007824520949], [-2.3455658060667703, -0.3084448591102636], [-0.3531917004907947, 2.7422494269413327], [0.273080856873386, 2.956083462929946]],
# [[-0.20808670832239384, 0.29087382886628127, 0.3928415162593641, -0.06277084529332996]],
# [[0.04421708865398105, 0.5706090967931113]]]

        # self.weights1 = w[0]
        # self.weights2 = w[1]
        # self.biases1 = w[2]
        # self.biases2 = w[3]

    def reset(self):
        self.x = initX
        self.y = initY
        self.vel = initVel
        self.angle = initAngle
        self.distanceTravelled = 0

    def move(self, steer, pedal):
        self.angle += steer * rotSpeed
        self.vel += pedal * acceleration
        self.vel = max(min_speed, min(self.vel, max_speed))

        self.distanceTravelled += abs(self.vel)

        self.x += math.cos(self.angle) * self.vel
        self.y -= math.sin(self.angle) * self.vel  # Pygame’s y-axis is inverted

    def isOnTrack(self, track_surface):
        if self.pixel_exists(self.x, self.y):
            color = track_surface.get_at((int(self.x), int(self.y)))
            return color == track_color  # White means off-track
        else:
            return False
    
    def draw(self, surface):
        rect_surface = pygame.Surface((15, 6), pygame.SRCALPHA)

        # Calculate color based on speed (normalized)
        normalized_speed = self.vel / max_speed  # Assuming self.vel and max_speed are defined
        normalized_speed = max(0, min(1, normalized_speed)) #Clamp the value between 0 and 1
        
        # Color gradient: Black -> Purple -> Red
        if normalized_speed < 0.5:
            red = int(255 * (normalized_speed * 2))  # Red increases linearly
            blue = int(255 * (normalized_speed * 2))
            purple = (red, 0, blue)
            rect_surface.fill(purple)
        else:
            red = 255
            blue = int(255 * (1 - (normalized_speed - 0.5) * 2))  # Blue decreases linearly
            purple = (red, 0, blue)
            rect_surface.fill(purple)
        

        rotated_surface = pygame.transform.rotate(rect_surface, self.angle * 180 / math.pi)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))

        surface.blit(rotated_surface, rotated_rect.topleft)

    def sensorReadings(self, track_surface):
        maximum = max_sensor
        sensors = []

        angles = [0, math.pi/2, -math.pi/2, math.pi/4, -math.pi/4]
        for angle in angles:
            sensor = maximum
            for distance in range(maximum):
                x = self.x + (math.cos(angle + self.angle) * distance)
                y = self.y - (math.sin(angle + self.angle) * distance)
                # Check if (x, y) is within bounds
                if self.pixel_exists(x, y):
                    color = track_surface.get_at((int(x), int(y)))
                    if color != track_color:
                        sensor = distance
                        break
                else:
                    sensor = maximum
                    break
            sensors.append(sensor / maximum)
        return sensors

        #DRAW ON THE LINES
    def draw_lines(self, sensors, track_surface):
        maximum = max_sensor
        angles = [0, math.pi/2, -math.pi/2, math.pi/4, -math.pi/4]
        for i in range(len(sensors)):
            angle = angles[i] + self.angle
            x = self.x + (math.cos(angle) * sensors[i] * maximum)
            y = self.y - (math.sin(angle) * sensors[i] * maximum)

            # Draw the sensor line.  Experiment with different colors.
            pygame.draw.line(track_surface, (255, 0, 0), (int(self.x), int(self.y)), (int(x), int(y)), 2)  # Red lines, 2 pixels thick
    
    def pixel_exists(self, x, y):
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            return True
        else:
            return False

    def getFitness(self):
        fitness = self.distanceTravelled
        return fitness
    
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def mutate(self):
        global mutation_frequency
        # Apply mutation to weights1
        mutation_frequency = max(0.75, 3 / (len(record) + 7)) #or just keep it at 0.2
        r = mutation
        all_mutators = [self.weights1, self.weights2, self.biases1, self.biases2]

        for mutator in all_mutators: #Mutate all weights and biases
                for i in range(len(mutator)):
                    for j in range(len(mutator[i])):
                        if random.random() < mutation_frequency:
                            mutator[i][j] += random.uniform(-r, r) 
        # if random.random() < speciation_rate: #NEW SPECIES
        #   
    
    # def relu(self, x):
    #     return np.maximum(0, x)

    def think(self, inputs):
        hidden_layer = tanh(dot(inputs, self.weights1) + self.biases1)  #HIDDEN
        # hidden_layer2 = tanh(dot(hidden_layer, self.weight2) + self.biases[1])  #HIDDEN
        outputs = tanh(dot(hidden_layer, self.weights2) + self.biases2)  #OUTPUT
        return outputs.flatten()

def create_oval_track(wall_thickness, straight_length, corner_radius):
    global initX, initY, initAngle, track_color
    initX = WIDTH*0.2
    initY = HEIGHT* 0.27
    initAngle = 0

    track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    track_surface.fill(WHITE)  # Transparent background

    # Calculate track dimensions
    track_height = 1.4 * (corner_radius + wall_thickness)  # Height is now fixed
    track_width = 1.8 * (corner_radius + straight_length + wall_thickness)

    # Calculate center of the track (vertically)
    center_y = HEIGHT // 2

    # Calculate the x-coordinates for the left and right edges of the track
    left_x = WIDTH // 2 - track_width // 2
    right_x = WIDTH // 2 + track_width // 2


    # Draw the outer oval (rounded rectangle)
    pygame.draw.rect(track_surface, GRAY, 
                     (left_x, center_y - track_height // 2, 
                      track_width, track_height), 
                     border_radius=corner_radius)

    # Calculate inner oval dimensions (for the hole)
    inner_width = track_width - 2 * wall_thickness
    inner_height = track_height - 2 * wall_thickness

    # Draw the inner oval (hole)
    pygame.draw.rect(track_surface, WHITE,
                     (left_x + wall_thickness, center_y - inner_height // 2,
                      inner_width, inner_height),
                     border_radius=corner_radius)

    track_color = track_surface.get_at((int(initX), int(initY)))


    return track_surface


def createTrack(width, height):
    global initX, initY, initAngle, track_color
    initX = WIDTH*0.135
    initY = HEIGHT*0.5
    initAngle = math.pi/2
    
    track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    track_surface.fill(WHITE) 

    image = pygame.image.load('track.png').convert_alpha()  # Convert for transparency
    image = pygame.transform.scale(image, (width, height))

    track_surface.blit(image, (0, 0))

    track_color = track_surface.get_at((int(initX), int(initY)))

    return track_surface

def createTrack2(width, height):
    global initX, initY, initAngle, track_color
    initX = WIDTH*0.4
    initY = HEIGHT*0.1
    initAngle = 0
    
    track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    track_surface.fill(WHITE) 

    image = pygame.image.load('track6.png').convert_alpha()  # Convert for transparency
    image = pygame.transform.scale(image, (width, height))

    track_surface.blit(image, (0, 0))

    track_color = track_surface.get_at((int(initX), int(initY)))

    return track_surface
    

def crossover(parent1, parent2):
    child = Car()  # Create a new car (child)

    # Weights
    child.weights1 = (parent1.weights1 + parent2.weights1) / 2
    child.weights2 = (parent1.weights2 + parent2.weights2) / 2

    # Biases
    child.biases1 = (parent1.biases1 + parent2.biases1) / 2
    child.biases2 = (parent1.biases2 + parent2.biases2) / 2

    return child

def run():
    global tick, drawLines
    # Initialize pygame
    pygame.init()
    font = pygame.font.SysFont("Arial Black", 12)  # Customizable system font

    top = 0

    # Screen settings
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    wall_thickness = 50
    straight_length = 95  # Adjust length of straight sections
    corner_radius = 180       # Adjust radius of the turns

    # track_surface = create_oval_track(wall_thickness, straight_length, corner_radius)
    track_surface = createTrack(WIDTH, HEIGHT)

    cars = []
    for _ in range(ipopulation):
        cars.append(Car())
    for i in range(len(cars) - 1):
        cars[i].mutate()
    

    running = True
    while running:
        steps = 0
        carsOnTrack = population
        avg_text = font.render(f"Best: {top:.2f} | Pop: {population:.2f} | Itt: {len(record)} | M Rate {mutation_frequency:.2f} | Strength {mutation:.2f} | TICK {tick}", True, (130, 160, 255))  # Light Blue text


        while steps < total_steps and carsOnTrack > 0: # and steps < total_steps 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(BLACK)
            screen.blit(track_surface, (0, 0))
            
            
            carsOnTrack = 0
            # Check if all cars have finished moving
            for car in cars:
                if car.isOnTrack(track_surface):
                    normalized_speed = car.vel / max_speed  # Normalize to 0-1
                    normalized_speed = max(0, min(1, normalized_speed))  # Clamp between 0 and 1
                    readings = car.sensorReadings(track_surface)
                    inputs = readings + [normalized_speed]
                    directions = car.think(inputs)
                    car.move(directions[0], directions[1])
                    carsOnTrack += 1
                    if drawLines == True:
                        car.draw_lines(readings, screen)
                
                car.draw(screen)  # Draw each car
                screen.blit(avg_text, (10, 10))  # Position the text at (10, 10)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_DOWN]:  #  Down
                tick = 60
            if keys[pygame.K_UP]:  # Up
                tick = 500
            if keys[pygame.K_LEFT]:
                drawLines = False
            if keys[pygame.K_RIGHT]:
                drawLines = True
                    
            pygame.display.flip()

            steps += 1
            # Slow down only this phase
            clock.tick(tick)
        

        # *** FAST EVOLUTION PHASE ***

        rankedCars = sorted(cars, key=lambda car: car.getFitness(), reverse=True)
        top_performers = rankedCars[:population // 4]  # Keep top 50%

        top = top_performers[0].getFitness()
        record.append(top)
        print(top)

        # Create the next generation: Start with the top performers
        next_generation = deepcopy(top_performers)  # Directly copy the top 50%

        # Generate offspring by mutating the top performers
        index = 0
        while len(next_generation) < population:
            parent1 = top_performers[index % len(top_performers)] 
            # parent2 = random.choice(top_performers)  # Choose a strong parent
            # child = crossover(parent1, parent2)
            child = deepcopy(parent1)
            child.mutate()
            next_generation.append(child)
            index += 1

        cars = next_generation  # Replace the old population with the new one

        # Reset cars for the next generation (do this AFTER creating the next generation)
        for car in cars:
            car.reset()


    pygame.quit()
    print(top_performers[0].weights1)
    print(top_performers[0].weights2)
    print(top_performers[0].biases1)
    print(top_performers[0].biases2)



run()
x = []
for i in range(len(record)):
    x.append(i)

plt.plot(x, record, marker='o', linestyle='-', color='r', label='Best Result')  #Speed
plt.xlabel("Itterations")
plt.legend()  # Add a legend
plt.grid(True)  # Add a grid
plt.show()  # Display the graph