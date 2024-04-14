from gaussian_splatting import *


if __name__ == "__main__":


    W = int(979)  # 1957  # 979
    H = int(546)  # 1091  # 546
    xy = np.stack((np.meshgrid(np.linspace(0, 1.0, W), np.linspace(0, 1.0, H))), axis=2)
    z = 1 - xy[:,:, 0]**2 + xy[:,:, 1]**2
    xyz = np.dstack((xy, z))

    sh = np.array([1.772484,  -1.772484, -1.772484, 
            -0.2, -0.7, 1.0, 
            2., 0.1, 0.6, 
            0.6, 0.2, 0.5])

    color = sh2color(sh[np.newaxis, :], xyz.reshape(-1,3))
    image = color.reshape([H, W, 3])

    plt.imshow(image)

    plt.show()
