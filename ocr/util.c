#include "emscripten.h"

typedef unsigned char uint8;

void image_to_grayscale(int width, int height, int channel, uint8 *img, uint8 *result) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int k = 0; k < channel; k++)
                sum += img[(i * width + j) * channel + k];
            result[i * width + j] = sum / channel;
        }
}

EMSCRIPTEN_KEEPALIVE void rgb_to_grayscale(int width, int height, uint8 *img, uint8 *result) {
    image_to_grayscale(width, height, 3, img, result);
}

EMSCRIPTEN_KEEPALIVE void rgba_to_grayscale(int width, int height, uint8 *img, uint8 *result) {
    image_to_grayscale(width, height, 4, img, result);
}
