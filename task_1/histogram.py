import sys
import os.path
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

MAX_WIDTH = 600
MAX_HEIGHT = 400

HISTOGRAM_WIDTH = 600
HISTOGRAM_HEIGHT = 400
HISTOGRAM_MARGIN = 20

if len(sys.argv) < 2:
    print "Usage: python histogram.py IMAGE_PATH"
    sys.exit(1)

image_path = sys.argv[1]
image_ext = os.path.splitext(image_path)[1]

if image_ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
    print "Invalid image format. Use one of these format: .jpg, .jpeg, .png, .bmp"
    sys.exit(1)

try:
    base_image = Image.open(image_path)
    width, height = base_image.size
    if width > MAX_WIDTH:
        height, width = height * MAX_WIDTH / width, MAX_WIDTH
    if height > MAX_HEIGHT:
        width, height = width * MAX_HEIGHT / height, MAX_HEIGHT
    base_image = base_image.resize((width, height))
except Exception as e:
    print "Cannot open image"
    sys.exit(1)

# ===================
img = base_image.load()

print img[0,0]
# ===================

def calculate_image_histogram(image, channel_id=0):
    width, height = image.size
    img = image.load()
    histogram = [0 for i in range(256)]
    for i in range(height):
        for j in range(width):
            color = img[j,i]
            color_sum = []
            if channel_id & 1:
                color_sum += [color[0]]
            if channel_id & 2:
                color_sum += [color[1]]
            if channel_id & 4:
                color_sum += [color[2]]
            value = round(sum(color_sum) / len(color_sum))
            histogram[int(value)] += 1
    
    return [float(x)/float(width * height) for x in histogram]


plt.plot(calculate_image_histogram(base_image, 1), 'r')
plt.plot(calculate_image_histogram(base_image, 2), 'g')
plt.plot(calculate_image_histogram(base_image, 4), 'b')
plt.plot(calculate_image_histogram(base_image, 7), 'k')

base_image.show("Base Image")
plt.show()
