# self-driving-car-neural-network
This project is a self-driving car simulator built entirely with NumPy and PyGame. Each car is controlled by a small neural network that takes in five distance sensors plus its current speed and outputs a vector telling the car steering and acceleration commands (Use the left/right arrow keys to see a visualization of what the cars can "see"). The cars improve through a natural-selection-style evolutionary algorithm that evolves their weights and biases over generations of 60 cars each. The system includes variable mutation rates, crossover, sensor visualization, dynamic speed-based rendering, and custom track generators. After enough iterations, the evolved behavior starts to resemble a tiny F1 race simulation, very cool to see!

(Open the NaturalSelectionCar.py file)
Controls
 - Left / Right Arrow = Toggle sensor visualization
 - Up Arrow = Increase simulation speed
 - Down Arrow = Decrease simulation speed

How It Works:

Sensors:
(Left and Right arrows toggle sensor visualization)
Each car has 5 rays (front, left, right, and two diagonals) plus its current speed, normalized to 0–1.

Neural Network:
A small feed-forward NN (6 → 4 → 2) outputs:

Evolutionary Loop After each generation:
 - Cars are ranked by distance traveled
 - The top performers are kept (top 50%)
 - Offspring are generated with mutation and optional crossover
 - Cars reset and the next generation begins
