import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd

# Sample data (replace with your actual data)
x_values_nr=5000
x_values = np.random.rand(x_values_nr)
y_values = np.random.rand(x_values_nr)
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg', 'path/to/image4.jpg', 'path/to/image5.jpg', 'path/to/image6.jpg', 'path/to/image7.jpg', 'path/to/image8.jpg', 'path/to/image9.jpg', 'path/to/image10.jpg']
df = pd.read_csv('/home/username/open_clip/finetuning/train_va_dataset_new.csv')
images_list=df['image_path'].tolist()
image_path=images_list[0].replace("/work/","/data/work-gcp-europe-west4-a/")
image_paths=[i.replace("/work/username/","/scratch/username/") for i in images_list[:x_values_nr]]

# Create a scatter plot
fig, ax = plt.subplots()
ax.scatter(x_values, y_values)
# Remove axes and labels
ax.axis('off')

# Remove tick marks
ax.set_xticks([])
ax.set_yticks([])

# Display images at each point
for x, y, image_path in zip(x_values, y_values, image_paths):
    image = plt.imread(image_path)
    imagebox = OffsetImage(image, zoom=0.1)  # You can adjust the zoom factor
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
    ax.add_artist(ab)

plt.savefig("/home/username/open_clip/finetuning/bokeh/test.png")