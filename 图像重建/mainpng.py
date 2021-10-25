from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
import random

'''
# 图片路径
img_path = 'A.png'
#img_path = 'A_save_img.png'
# 以 BGR 方式读取图片
img = cv2.imread(img_path)
#彩色图像使用 OpenCV 加载时是 BGR 模式，但是 Matplotlib 是 RGB 模式。
#所以彩色图像如果已经被 OpenCV 读取，那它将不会被 Matplotlib 正确显示。因此我们将 BGR模式转换为 RGB 模式即可。
# 将 BGR 方式转换为 RGB 方式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 打印图片类型
print(type(img))
print(img)
# 关闭坐标轴
plt.axis('off')
# 展示图片
plt.imshow(img)
plt.show()
'''

def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()

'''
OpenCV 保存一个图片使用函数 cv2.imwrite(filename, img[, params])：
filename：保存文件路径及文件名，文件名要加格式
img：需要保存的图片
下面我们用 cv2.imwrite() 来封装一个保存图片的函数。
'''
def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)

    # 图像数组数据放缩在 0-1 之间
    # numpy中的astype()函数可用于转化dateframe某一行的数据类型
    return image.astype(np.double) / info.max


def noise_mask_image(img, noise_ratio=[0.8,0.4,0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    #创建图片副本，对副本的操作不会影响原本
    noise_img = np.copy(img)
    #分别对三个通道添加噪声
    for i in range(3):
        for j in range(img.shape[0]):
            #将下标包装成列表，用于取样
            masks = list(range(img.shape[1]))
            #使用random.sample()函数随机取样选择需要加噪声的像素点，选取数量由相应通道的噪声比例决定
            masks = random.sample(masks, int(img.shape[1]*noise_ratio[i]))
            #按照取样得到的样本添加噪声
            for k in range(img.shape[1]):
                if k in masks:
                    noise_img[j,k,i] = 0
    #图像矩阵值 0-1 之间
    noise_img[noise_img > 1.0] = 1.0
    noise_img[noise_img < 0.0] = 0.0
    # -----------------------------------------------
    return noise_img

def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像
    :param img:原始图像
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0

    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)

    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None

    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))

    return round(error, 3)


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    #按照cutsize分割图像
    cutsize=10
    #获取ndarray的shape
    rows,cols,channel=res_img.shape
    #分别按照cols和rows进行分割
    row_cut=rows//cutsize
    col_cut=cols//cutsize
    #按通道进行预测恢复，三通道即依次恢复三次
    for ind_ch in range(channel):
        for row in range(row_cut + 1):
            #得到每一块的起始索引
            row_base = row * cutsize
            if row == row_cut:
                #给最后剩余不够cutsize的地方补上一些像素，构造成cutsize大小
                row_base = rows - cutsize
            for col in range(col_cut + 1):
                #同上
                col_base = col * cutsize
                if col == col_cut:
                    col_base = cols - cutsize
                #训练集的特征值
                x_train = []
                #训练集的目标值
                y_train = []
                #测试集的特征值
                x_test = []
                #以块为单位，获取训练集
                for i in range(row_base, row_base + cutsize):
                    for j in range(col_base, col_base + cutsize):
                        if noise_mask[i, j, ind_ch] == 0:
                            #注意这里为什么将0的值作为判断是噪声点的依据，其实是有原因的，严格来说，为0并不代表一定是噪声点，但
                            #考虑两个原因，一是如果该像素点的rgb中的某个值本来就为0，那么说明大概率情况下周围的点也是这种分布，
                            # 因此可以通过模型正确预测出该点值为0，不会对结果有较大影响，二是如果对所有像素点进行预测,代价比较大
                            x_test.append([i, j])
                            continue
                        x_train.append([i, j])
                        y_train.append([res_img[i, j, ind_ch]])
                if x_train == []:
                    print("x_train is None")
                    continue
                #实例化一个线性回归预测器
                LR = LinearRegression()
                #进行有监督学习
                LR.fit(x_train, y_train)
                #进行预测
                predict = LR.predict(x_test)
                for i in range(len(x_test)):
                    #通过预测进行单通道降噪
                    res_img[x_test[i][0], x_test[i][1], ind_ch] = predict[i][0]
    #图像矩阵值 0-1 之间
    res_img[res_img > 1.0] = 1.0
    res_img[res_img < 0.0] = 0.0
    # ---------------------------------------------------------------
    return res_img


# 原始图片
# 加载图片的路径和名称
img_path = 'E.jpg'

# 读取原始图片
img = read_image(img_path)

# 展示原始图片
plot_image(image=img, image_title="original image")

# 生成受损图片
# 图像数据归一化
nor_img = normalization(img)

# 每个通道数不同的噪声比率
noise_ratio = [0.4, 0.6, 0.8]

# 生成受损图片
noise_img = noise_mask_image(nor_img, noise_ratio)

if noise_img is not None:
    # 展示受损图片
    plot_image(image=noise_img, image_title="the noise_ratio = %s of original image" % noise_ratio)

    # 恢复图片
    res_img = restore_image(noise_img)

    # 计算恢复图片与原始图片的误差
    print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))

    # 展示恢复图片
    plot_image(image=res_img, image_title="restore image")

    # 保存恢复图片
    save_image('res_' + img_path, res_img)
else:
    # 未生成受损图片
    print("返回值是 None, 请生成受损图片并返回!")