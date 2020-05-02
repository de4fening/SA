import numpy as np
from functools import reduce
from scipy.stats.distributions import chi2
from scipy.stats import t


def file_parsing(file_name, first_attribute_number, second_attribute_number):
    first_attributes = []
    second_attributes = []
    with open(file_name, 'r') as file:
        for line in file:
            parsed_data = line.split()
            if parsed_data[first_attribute_number] != '?':
                if parsed_data[second_attribute_number] != '?':
                    first_attributes.append(float(parsed_data[first_attribute_number]))
                    second_attributes.append(float(parsed_data[second_attribute_number]))
    return first_attributes, second_attributes


def expected_value(variable):
    return reduce((lambda x, y: x + y), variable) / len(variable)


def variance(variable):
    return expected_value(np.square(variable)) - pow(expected_value(variable), 2)


def displaced_variance(variable):
    n = len(variable)
    return variance(variable) * n / (n - 1)


def confidence_interval(variable, alpha):
    coefficient = np.abs(t.ppf(alpha / 2, df=(len(variable) - 1)))
    return coefficient * np.sqrt(displaced_variance(variable) / len(variable))


def interval_border(variable, alpha):
    length = len(variable)
    high = chi2.ppf(alpha / 2, df=(length - 1))
    low = chi2.ppf(1 - alpha / 2, df=(length - 1))
    variable_displaced_variance = displaced_variance(variable)
    low_border = variable_displaced_variance * (length - 1) / low
    high_border = variable_displaced_variance * (length - 1) / high
    return low_border, high_border


def check_hypothesis_with_variance(first_expected, second_expected, first_variance,
                                   second_variance, first_amount, second_amount):
    return np.abs(first_expected - second_expected) / np.sqrt((first_variance / len(first_amount))
                                                              + (second_variance / len(second_amount)))


def check_hypothesis_without_variance(first_expected, second_expected, first_displaced_variance,
                                      second_displaced_variance, first_amount, second_amount):
    return np.abs(first_expected - second_expected) / np.sqrt(
        (len(first_amount) - 1) * first_displaced_variance + (len(second_amount) - 1) * second_displaced_variance) \
           * np.sqrt(((len(first_amount) * len(second_amount)) * (len(first_amount) + len(second_amount) - 2)) / (
            len(first_amount) + len(second_amount)))


def main():
    RI, AI = file_parsing('11-glass.txt', 1, 4)
    RI = list(map(float, RI))
    AI = list(map(float, AI))

    alpha = 0.05

    expected_value_RI = expected_value(RI)
    expected_value_AI = expected_value(AI)
    RI_variance = variance(RI)
    AI_variance = variance(AI)
    RI_displaced_variance = displaced_variance(RI)
    AI_displaced_variance = displaced_variance(AI)
    RI_standard_deviation = np.sqrt(variance(RI))
    AI_standard_deviation = np.sqrt(variance(AI))

    print('Expected value of average RI = ', expected_value_RI)
    print('Expected value of average AI = ', expected_value_AI)
    print('Variance value of average RI = ', RI_variance)
    print('Variance value of average AI = ', AI_variance)
    print('Displaced variance value of average RI = ', RI_displaced_variance)
    print('Displaced variance value of average AI = ', AI_displaced_variance)
    print('Standard deviation value of average RI = ', RI_standard_deviation)
    print('Standard deviation value of average AI = ', AI_standard_deviation)

    print('Confidence interval of average RI expected value: {} < E(x) < {}'.format(
        expected_value_RI - confidence_interval(RI, alpha),
        expected_value_RI + confidence_interval(RI, alpha)
    ))

    print('Confidence interval of average AI expected value: {} < E(x) < {}'.format(
        expected_value_AI - confidence_interval(AI, alpha),
        expected_value_AI + confidence_interval(AI, alpha)
    ))

    RI_low_border, AI_high_border = interval_border(RI, alpha)
    print('Confidence interval of average RI variance: {} < sigma^2 < {}'.format(
        RI_low_border,
        AI_high_border
    ))

    AI_low_border, AI_high_border = interval_border(AI, alpha)
    print('Confidence interval of average AI variance: {} < sigma^2 < {}'.format(
        AI_low_border,
        AI_high_border
    ))

    print('Check hypothesis with know variance: ', check_hypothesis_with_variance(
        expected_value_RI, expected_value_AI, RI_variance,
        AI_variance, RI, AI))
    print('Check hypothesis with unknown variance: ', check_hypothesis_without_variance(
        expected_value_RI, expected_value_AI, RI_displaced_variance,
        AI_displaced_variance, RI, AI))


if __name__ == '__main__':
    main()
