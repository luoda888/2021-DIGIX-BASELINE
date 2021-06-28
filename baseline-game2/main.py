from Preprocess import preprocess
from Build_PU_data import build_pu_data
from Train_Bert import train_bert
from Train_PU_model import train_pu_model
from Joint_Predictor import joint_predictor

if __name__ == "__main__":
    # 数据集预处理：对原始训练集，测试集进行简单的清洗，从训练集中输出 带标记样本集（P）和 未知样本集（U）
    preprocess()
    # 使用 P 训练 Bert分类器（10分类）
    train_bert()
    # 使用已训练的 Bert分类器预测 U，并输出 可靠负样本集（RN）
    build_pu_data()
    # 使用 P 和 RN 训练 二分类器
    train_pu_model()
    # 使用 Bert分类器 和 二分类器 联合预测 测试集上样本的类别，并格式化输出结果至文件
    joint_predictor()