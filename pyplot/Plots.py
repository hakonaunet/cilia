import os
import random
import matplotlib.pyplot as plt

def plot_test_1(x, y, z, measurement_points, filename):
    plt.figure()  # Create a new figure
    plt.plot(measurement_points, x, label='x-component')
    plt.plot(measurement_points, y, label='y-component')
    plt.plot(measurement_points, z, label='z-component')
    plt.xlabel("Measurement points along the z-axis")  # Set x-axis title
    plt.ylabel("Velocity field")  # Set y-axis title
    plt.title("Test 1: symmetric velocity field around cilia plane")  # Set plot title
    plt.legend()  # Add a legend
    plt.grid(True)  # Add grid

    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    plt.savefig(os.path.join(output_dir, filename))  # Save the figure to the output directory
    plt.close()  # Close the figure to free up memory

def plot_data(x, y, filename):
    plt.figure()  # Create a new figure
    plt.plot(x, y)
    plt.xlabel("Time (s)")  # Set x-axis title
    plt.ylabel("Order Parameter")  # Set y-axis title
    plt.title("Kuramoto Order Parameter")  # Set plot title
    plt.yticks([0, 1])  # Set y-axis ticks

    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    plt.savefig(os.path.join(output_dir, filename))  # Save the figure to the output directory
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    measurement_points = list(range(31))  # x values from 0 to 30
    x = [random.random() for _ in measurement_points]  # Random y values
    y = [random.random() for _ in measurement_points]  # Random y values
    z = [random.random() for _ in measurement_points]  # Random y values
    
    plot_test_1(x, y, z, measurement_points, "Test1_test_plot.png")

    x = list(range(31))  # x values from 0 to 30
    y = [random.random() for _ in x]  # Random y values
    plot_data(x, y, "order_parameter_test_plot.png")