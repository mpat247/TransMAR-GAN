import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to the test results directory
test_results_dir = './test_results'

# List all files in the directory
result_files = os.listdir(test_results_dir)

# Assuming you want to plot the first result (you can adjust the index as needed)
if result_files:
    result_path = os.path.join(test_results_dir, result_files[0])
    
    # Open the image and display it
    img = Image.open(result_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.title("Generated CT Image Result")
    plt.show()
else:
    print("No results found in the test_results directory.")

