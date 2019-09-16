import math
def cal_mean(readings):
    """
    Function to calculate the mean value of the input readings
    :param readings:
    :return:
    """
    readings_total = sum(readings)
    number_of_readings = len(readings)
    mean = readings_total / float(number_of_readings)
    return mean

def cal_variance(readings):
    """
    Calculating the variance of the readings
    :param readings:
    :return:
    """

    # To calculate the variance we need the mean value
    # Calculating the mean value from the cal_mean function
    readings_mean = cal_mean(readings)
    # mean difference squared readings
    mean_difference_squared_readings = [pow((reading - readings_mean), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)


def cal_covariance(readings_1, readings_2):
    """
    Calculate the covariance between two different list of readings
    :param readings_1:
    :param readings_2:
    :return:
    """
    readings_1_mean = cal_mean(readings_1)
    readings_2_mean = cal_mean(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in range(0, readings_size):
        covariance += (readings_1[i] - readings_1_mean) * (readings_2[i] - readings_2_mean)
    return covariance / float(readings_size - 1)


def cal_simple_linear_regression_coefficients(x_readings, y_readings):
    """
    Calculating the simple linear regression coefficients (B0, B1)
    :param x_readings:
    :param y_readings:
    :return:
    """
    
    # Coefficient B1 = covariance of x_readings and y_readings divided by variance of x_readings
    # Directly calling the implemented covariance and the variance functions
    # To calculate the coefficient B1
    if cal_variance(x_readings) != 0:
        b1 = cal_covariance(x_readings, y_readings) / float(cal_variance(x_readings))
        # Coefficient B0 = mean of y_readings - ( B1 * the mean of the x_readings )
        b3 = cal_mean(y_readings) - (b1 * cal_mean(x_readings))
        b2 = -1
    else:
        b1 = 1
        b2 = 0
        b3 = -x_readings[0]
    return b1, b2, b3
    
def find_projection(a, b, c, x, y):
    # ax+by+c = 0
    m = math.sqrt(b * b + a * a)
    x0 = round(x - a * (a * x + b * y + c) / m / m, 0)
    y0 = round(y - b * (a * x + b * y + c) / m / m, 0)
    return x0, y0
