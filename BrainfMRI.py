import numpy as np
import scipy.io
import pandas as pd

"""a) For each i, j, compute the correlation between xij and µ. This should produce a 64 × 64 matrix C
indicating the correlation of each pixel of the brain with the thumb movement. Display C as an image."""
mat = scipy.io.loadmat('thumb_data.mat')
x_mat = mat.get('X')  # 64x64x122
mu_vect = mat.get('mu')  # 1x122
mat_size = 64
ans_mat = np.zeros(shape=(64, 64), dtype=object)
for row in range(mat_size):
    for column in range(mat_size):
        time = x_mat[row][column]
        corr = np.corrcoef(np.transpose(time), np.transpose(mu_vect))
        ans_mat[row][column] = corr[0][1]

dataFrame = pd.DataFrame(ans_mat)
dataFrame = dataFrame.fillna(0)

# Output to an image
import scipy.misc

scipy.misc.imsave('Image_org.jpg', dataFrame.values)

"""b) We are taking the absolute value of """


def retActivationMap(tau):
    activation_map = np.zeros(shape=(64, 64))
    for row in range(len(dataFrame)):
        for column in range(len(dataFrame[row])):
            if dataFrame[row][column] < tau:
                activation_map[row][column] = 1
    return activation_map


mean = (np.mean(abs(dataFrame.values)))
# print(mean)
activation_map = retActivationMap(mean)

activation_arr = np.zeros(shape=5, dtype=object)
for x in range(5):
    tau = mean + .1 * x
    print('tau value = ', tau, end=': ')
    activation_arr[x] = retActivationMap(abs(tau))
    image_name = 'Image_act' + str(x) + '.jpg'
    print(image_name)
    scipy.misc.imsave(image_name, np.transpose(activation_arr[x]))
