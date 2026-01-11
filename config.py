"""
系统配置文件
定义AI机器人的各项配置参数
"""

class Config:
    """AI机器人系统配置"""
    
    # BERT模型配置
    BERT_MODEL_NAME = 'bert-base-chinese'  # 中文BERT模型
    MAX_SEQ_LENGTH = 512  # 最大序列长度
    
    # 神经网络配置
    INPUT_SIZE = 768  # BERT输出维度
    HIDDEN_SIZE = 512  # 隐藏层大小
    OUTPUT_SIZE = 768  # 输出维度
    NUM_LAYERS = 3  # 神经网络层数
    DROPOUT_RATE = 0.2  # Dropout比率
    
    # 训练配置
    LEARNING_RATE = 0.001  # 学习率
    BATCH_SIZE = 32  # 批处理大小
    OPTIMIZER_TYPE = 'adam'  # 优化器类型
    
    # 搜索配置
    MAX_SEARCH_RESULTS = 5  # 最大搜索结果数
    SEARCH_TIMEOUT = 10  # 搜索超时时间（秒）
    
    # 决策模型配置
    DECISION_TREE_MAX_DEPTH = 10  # 决策树最大深度
    RANDOM_FOREST_N_ESTIMATORS = 10  # 随机森林估计器数量
    
    # 数据分析配置
    MIN_DATA_POINTS_FOR_ANALYSIS = 5  # 分析所需的最小数据点
    ANALYSIS_SAMPLE_SIZE = 100  # 分析样本大小
    
    # 系统配置
    RESPONSE_TIMEOUT = 30  # 响应超时时间（秒）
    MAX_HISTORY_SIZE = 100  # 最大历史记录大小
    LOG_LEVEL = 'INFO'  # 日志级别
    
    # 用户界面配置
    CHAT_HISTORY_LIMIT = 50  # 聊天历史限制
    AUTO_SAVE_INTERVAL = 300  # 自动保存间隔（秒）
    
    @classmethod
    def get_model_config(cls):
        """获取模型相关配置"""
        return {
            'input_size': cls.INPUT_SIZE,
            'hidden_size': cls.HIDDEN_SIZE,
            'output_size': cls.OUTPUT_SIZE,
            'num_layers': cls.NUM_LAYERS,
            'dropout_rate': cls.DROPOUT_RATE
        }
    
    @classmethod
    def get_training_config(cls):
        """获取训练相关配置"""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'batch_size': cls.BATCH_SIZE,
            'optimizer_type': cls.OPTIMIZER_TYPE
        }
    
    @classmethod
    def get_search_config(cls):
        """获取搜索相关配置"""
        return {
            'max_results': cls.MAX_SEARCH_RESULTS,
            'timeout': cls.SEARCH_TIMEOUT
        }


# 默认配置实例
DEFAULT_CONFIG = Config()

# 开发环境配置
class DevConfig(Config):
    """开发环境配置"""
    LOG_LEVEL = 'DEBUG'
    MAX_SEARCH_RESULTS = 3
    AUTO_SAVE_INTERVAL = 60


# 生产环境配置
class ProdConfig(Config):
    """生产环境配置"""
    LOG_LEVEL = 'WARNING'
    MAX_SEARCH_RESULTS = 10
    RESPONSE_TIMEOUT = 15


# 测试环境配置
class TestConfig(Config):
    """测试环境配置"""
    MAX_SEARCH_RESULTS = 2
    MIN_DATA_POINTS_FOR_ANALYSIS = 2
    LOG_LEVEL = 'ERROR'
