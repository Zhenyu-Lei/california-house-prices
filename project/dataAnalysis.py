from torch import nn
from myUtils import readData
import pandas as pd


def main():
    # 读取数据
    train_data, test_data = readData('./data/train.csv', './data/test.csv')

    object_features = train_data.dtypes[train_data.dtypes == 'object'].index

    # 对数值对象进行相关性分析
    # correlation(train_data, "./data")
    # 根据相关矩阵，相关系数绝对值较大的特征有：
    numeric_main_features = ['Bathrooms', 'Full bathrooms', 'Tax assessed value', 'Annual tax amount', 'Listed Price',
                             'Last Sold Price']
    # 对object对象(离散值)进行分析
    # pd.set_option('display.max_columns', 1000)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.max_colwidth', 1000)
    # train_object = train_data[object_features]
    # print(train_object.describe())

    object_main_features = ['Type', 'Cooling', 'Bedrooms', 'Region', 'Middle School', 'High School', 'Cooling features',
                            'City', 'State']
    all_features_list = numeric_main_features + object_main_features
    # 合并特征
    all_features = pd.concat((train_data[all_features_list], test_data[all_features_list]))
    print(all_features.dtypes)

    # 将缺失值变为均值
    # 将特征缩放到0均值和单位方差来归一化
    # 将非object数据变为均值为0，方差为1
    all_features[numeric_main_features] = all_features[numeric_main_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # 将数值数据中not number的数据用0填充(其实是均值，修改了分布均值为0)
    all_features[numeric_main_features] = all_features[numeric_main_features].fillna(0)

    # 处理离散值，用独热编码替换它们
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # print(all_features[:4].values)
    all_features.to_csv("./data/oneHot_all_feature.csv", index=False, encoding="utf-8")

    print("数据生成完成")


if __name__ == '__main__':
    main()
