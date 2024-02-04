import os
import random
import matplotlib.pyplot as plt

def plot_data(x, y, z, measurement_points, filename):
    plt.figure()  # Create a new figure
    plt.plot(measurement_points, x, label='x-component')
    plt.plot(measurement_points, y, label='y-component')
    plt.plot(measurement_points, z, label='z-component')
    plt.xlabel("Measurement points along the z-axis")  # Set x-axis title
    plt.ylabel("Velocity field")  # Set y-axis title
    plt.title("Test 1: symmetric velocity field around cilia plane")  # Set plot title
    plt.legend()  # Add a legend

    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    plt.savefig(os.path.join(output_dir, filename))  # Save the figure to the output directory
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    x = list(range(31))  # x values from 0 to 30
    y = [random.random() for _ in x]  # Random y values
    plot_data(x, y, "Test1_test_plot.png")