import os
import random
import matplotlib.pyplot as plt

def plot_data(x, y, filename):
    plt.figure()  # Create a new figure
    plt.plot(x, y)
    plt.xlabel("Time (s)")  # Set x-axis title
    plt.ylabel("Order Parameter")  # Set y-axis title
    plt.title("Kuramoto Order Parameter")  # Set plot title
    plt.yticks([0, 1])  # Set y-axis ticks

    output_dir = "file_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    plt.savefig(os.path.join(output_dir, filename))  # Save the figure to the output directory
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    x = list(range(31))  # x values from 0 to 30
    y = [random.random() for _ in x]  # Random y values
    plot_data(x, y, "order_parameter_test_plot.png")