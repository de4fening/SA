from PIL import Image
from pylab import *
import scipy.stats as st
import numpy as np

file_path = 'resources/'


# Transform RGB image to gray_scale image
def img_transform(img_name):
    image = Image.open(file_path + img_name).convert('L')  # Открываем изображение и конвертируем в полутоновое
    image.save(file_path + "gray_scale_" + img_name)
    image.show()
    return image


# Image histogram and statistic
def img_histogram(image):
    img_array = array(image)
    hist_data = img_array.flatten()

    figure()
    hist(hist_data, 32)
    show()

    stat = [np.mean(hist_data), np.std(hist_data, ddof=1), st.mode(hist_data)[0][0], np.median(hist_data)]
    return hist_data, stat


# Hypothesis check
def hyp_check(data_set):
    chi2, p = st.chisquare(data_set)
    message = "Test Statistic: {}\np-value: {}"
    print(message.format(chi2, p))


def main():
    first_image = img_transform("first_image.jpg")
    second_image = img_transform("second_image.jpg")

    first_hist, first_hist_stat = img_histogram(first_image)
    second_hist, second_hist_stat = img_histogram(second_image)
    print('|Average|   |rms|   |mode|   |median|')
    print("First image: ", first_hist_stat)
    print("Second image: ", second_hist_stat)

    hist_correlation_coefficient = np.corrcoef(np.asarray(first_hist).flatten(),
                                               np.asarray(second_hist).flatten())[1, 0]
    print("Histogram correlation: ", hist_correlation_coefficient)

    image_correlation = np.corrcoef(np.asarray(first_image).flatten(), np.asarray(second_image).flatten())[1, 0]
    print('Images correlation', image_correlation)

    print('\nHypothesis:\n')
    first = [0.92, 1.4, 13.3, 16.9, 0.98, 1.1]
    second = [2.5, 6.5, 9.6, 7.9, 5, 3]
    hyp_check(first)
    hyp_check(second)


if __name__ == "__main__":
    main()
