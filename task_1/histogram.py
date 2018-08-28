import sys
import os.path

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
    # import random
    return [i/255.0 for i in range(256)]


def generate_histogram_image(histogram, color=(255, 0, 0)):
    histogram_points = [
        (
            HISTOGRAM_MARGIN + ((HISTOGRAM_WIDTH - 2 * HISTOGRAM_MARGIN) / 256.0) * i,
            HISTOGRAM_HEIGHT - HISTOGRAM_MARGIN - (HISTOGRAM_HEIGHT - 2 * HISTOGRAM_MARGIN) * value
        )
        for i, value in enumerate(histogram)
    ]
    histogram_image = Image.new('RGB', (HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT), color=(255, 255, 255))
    histogram_draw = ImageDraw.Draw(histogram_image)
    histogram_draw.line(histogram_points, fill=color)
    return histogram_image


red_histogram_image = generate_histogram_image(calculate_image_histogram(base_image, 1), (255, 0, 0))
green_histogram_image = generate_histogram_image(calculate_image_histogram(base_image, 2), (0, 255, 0))
blue_histogram_image = generate_histogram_image(calculate_image_histogram(base_image, 4), (0, 0, 255))
grayscale_histogram_image = generate_histogram_image(calculate_image_histogram(base_image, 7), (0, 0, 0))

base_image.show("Base Image")
red_histogram_image.show("Red Histogram")
green_histogram_image.show("Green Histogram")
blue_histogram_image.show("Blue Histogram")
grayscale_histogram_image.show("Grayscale Histogram")
