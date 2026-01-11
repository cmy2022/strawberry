"""
简单测试脚本，验证AI机器人的基本功能
"""

from simple_neural_ai_robot import SimpleNeuralAIBot

def test_basic_functionality():
    print("测试AI机器人的基本功能...")
    
    # 创建机器人实例
    bot = SimpleNeuralAIBot()
    
    # 测试对话功能
    print("\n1. 测试对话功能:")
    response = bot.chat("你好，这是一个测试")
    print(response)
    
    # 测试搜索功能
    print("\n2. 测试搜索功能:")
    response = bot.chat("请搜索人工智能的信息")
    print(response)
    
    # 测试决策功能
    print("\n3. 测试决策功能:")
    response = bot.chat("我应该选择哪个深度学习框架？")
    print(response)
    
    # 测试指令执行功能
    print("\n4. 测试指令执行功能:")
    response = bot.chat("分析需求开发一个聊天机器人")
    print(response)
    
    print("\n所有基本功能测试完成！")

if __name__ == "__main__":
    test_basic_functionality()
