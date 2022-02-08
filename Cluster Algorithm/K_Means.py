import numpy as np
import cv2
from sklearn import metrics


def K_Means_OpenCV(k, img):
    # OpenCV实现K-means
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类 聚集成k类
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))

    # 图像转换为RGB显示
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst


from sklearn.cluster import KMeans
import PIL.Image as image


def K_Means_sklearn(k, img):
    # sklearn实现K-means
    # 将CV2打开的图像转换为PIL格式
    PIL_img = image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    row, col = PIL_img.size

    # img = img.convert('RGB')  # 转RGB 否则会出现int不可以迭代的错误 上一步已经包括了所以可以忽略
    print(PIL_img)
    data = []

    for x in range(row):
        for y in range(col):
            r, g, b = PIL_img.getpixel((x, y))
            data.append([r / 256.0, g / 256.0, b / 256.0])

    Data = np.mat(data)

    label = KMeans(n_clusters=k).fit_predict(Data)
    label = label.reshape([row, col])
    pic_new = image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))

    # 将PIL图像转换为CV2格式
    dst = cv2.cvtColor(np.asarray(label), cv2.COLOR_RGB2BGR)
    return dst


from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


def K_Means_sklearn_Auto(k_min, k_max, img):
    # sklearn实现K-means聚类，并利用calinski_harabasz_score计算聚类分数用于决定k值
    # 后用OpenCV输出最佳k值对应聚类结果
    PIL_img = image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    row, col = PIL_img.size

    # img = img.convert('RGB')  # 转RGB 否则会出现int不可以迭代的错误 上一步已经包括了所以可以忽略
    # print(img)
    data = []

    for x in range(row):
        for y in range(col):
            r, g, b = PIL_img.getpixel((x, y))
            data.append([r / 256.0, g / 256.0, b / 256.0])

    Data = np.mat(data)

    k_fine = k_min
    score_max = 0
    for k in range(k_min, k_max + 1):
        mid_result = KMeans(n_clusters=k).fit(Data)
        # 利用silhouette_score计算分数较为准确但占用内存，运算速度慢
        # score = silhouette_score(Data, mid_result.labels_, metric='euclidean', sample_size=len(Data))  # 计算得分
        # 利用calinski_harabasz_score计算分数很粗糙，耗时短
        score = calinski_harabasz_score(Data, mid_result.labels_)  # 计算得分
        print(score)
        if score > score_max:
            score_max = score
            k_fine = k
    return K_Means_OpenCV(k_fine, img), k_fine

def K_Means_OpenCV_Auto(k_min, k_max, img):
    # 尚未完工
    # OpenCV实现K-means聚类，并利用calinski_harabasz_score计算聚类分数用于决定k值
    # 后用OpenCV输出最佳k值对应聚类结果
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # scores = []
    k_fin = k_min
    score_max = 0
    for k in range(k_min, k_max + 1):
        # K-Means聚类 聚集成k类
        compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

        score = metrics.silhouette_score(data,labels,metric='euclidean',
                                         same_size=len(labels))
        print(score)
        if score > score_max:
            score_max = score
            k_fin = k
    return K_Means_OpenCV(k_fin, img), k_fin