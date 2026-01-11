"""
AI机器人使用示例
展示如何使用深度学习神经网络智能AI机器人的各种功能
"""

from neural_ai_robot import NeuralAIBot
from config import DEFAULT_CONFIG
import time
import json


def demo_basic_conversation():
    """基础对话功能演示"""
    print("="*60)
    print("基础对话功能演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    test_inputs = [
        "你好，我是张三。",
        "你能告诉我什么是人工智能吗？",
        "今天的天气怎么样？",
        "推荐一些好的编程书籍。",
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[{i}] 用户输入: {user_input}")
        response = bot.chat(user_input)
        print(f"机器人响应:\n{response}\n")
        print("-" * 50)


def demo_search_and_analysis():
    """搜索与分析功能演示"""
    print("="*60)
    print("搜索与分析功能演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    search_queries = [
        "搜索人工智能最新发展",
        "查找PyTorch教程",
        "了解机器学习应用案例",
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n[{i}] 搜索查询: {query}")
        response = bot.chat(query)
        print(f"搜索结果:\n{response}\n")
        print("-" * 50)


def demo_decision_support():
    """决策支持功能演示"""
    print("="*60)
    print("决策支持功能演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    decision_queries = [
        "我应该选择哪个深度学习框架？",
        "推荐一个适合初学者的编程语言",
        "如何选择数据库系统？",
    ]
    
    for i, query in enumerate(decision_queries, 1):
        print(f"\n[{i}] 决策请求: {query}")
        response = bot.chat(query)
        print(f"决策结果:\n{response}\n")
        print("-" * 50)


def demo_project_management():
    """项目管理功能演示"""
    print("="*60)
    print("项目管理功能演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    project_queries = [
        "分析开发一个聊天机器人的需求",
        "设计一个人工智能系统的架构",
        "制定一个移动应用的开发计划",
    ]
    
    for i, query in enumerate(project_queries, 1):
        print(f"\n[{i}] 项目请求: {query}")
        response = bot.chat(query)
        print(f"项目分析:\n{response}\n")
        print("-" * 50)


def demo_performance_test():
    """性能测试演示"""
    print("="*60)
    print("性能测试演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    # 测试响应时间
    test_inputs = [
        "简单问候",
        "复杂问题需要分析",
        "需要搜索的信息请求",
    ]
    
    total_time = 0
    for i, user_input in enumerate(test_inputs, 1):
        start_time = time.time()
        response = bot.chat(user_input)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        print(f"\n[{i}] 输入: {user_input}")
        print(f"响应时间: {elapsed_time:.2f}秒")
        print(f"响应片段: {response.split()[0:10]}...\n")
    
    avg_time = total_time / len(test_inputs)
    print(f"平均响应时间: {avg_time:.2f}秒")
    print(f"总处理时间: {total_time:.2f}秒")


def demo_self_optimization_tracking():
    """自我优化跟踪演示"""
    print("="*60)
    print("自我优化跟踪演示")
    print("="*60)
    
    bot = NeuralAIBot()
    
    print("初始优化状态:", bot.self_optimizer.training_history)
    
    # 进行多次交互触发自我优化
    interactions = [
        "第一次交互",
        "第二次交互", 
        "第三次交互",
        "第四次交互",
        "第五次交互",
        "第六次交互",
        "第七次交互",
        "第八次交互",
        "第九次交互",
        "第十次交互",
    ]
    
    for i, interaction in enumerate(interactions, 1):
        response = bot.chat(interaction)
        
        # 每10次交互后会触发优化
        if i % 10 == 0:
            print(f"\n第{i}次交互后优化状态:")
            print(f"优化历史长度: {len(bot.self_optimizer.training_history)}")
            if bot.self_optimizer.training_history:
                recent_losses = [record['loss'] for record in bot.self_optimizer.training_history[-3:]]
                print(f"最近3次优化损失: {recent_losses}")


def run_all_demos():
    """运行所有演示"""
    print("深度学习神经网络智能AI机器人演示程序")
    print("开始运行所有功能演示...\n")
    
    try:
        demo_basic_conversation()
        demo_search_and_analysis()
        demo_decision_support()
        demo_project_management()
        demo_performance_test()
        demo_self_optimization_tracking()
        
        print("\n" + "="*60)
        print("所有演示已完成!")
        print("="*60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_demo():
    """交互式演示"""
    print("="*60)
    print("交互式演示 - 深度学习神经网络智能AI机器人")
    print("="*60)
    print("输入特殊命令来体验不同功能:")
    print("- 'demo_all': 运行所有演示")
    print("- 'basic': 基础对话演示")
    print("- 'search': 搜索分析演示") 
    print("- 'decision': 决策支持演示")
    print("- 'project': 项目管理演示")
    print("- 'performance': 性能测试演示")
    print("- 'quit' 或 'exit': 退出程序")
    print("- 其他任意文本: 与机器人正常对话")
    print("="*60)
    
    bot = NeuralAIBot()
    
    while True:
        try:
            user_input = input("\n请输入您的命令或消息: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("感谢使用AI机器人演示程序，再见！")
                break
            elif user_input.lower() == 'demo_all':
                run_all_demos()
            elif user_input.lower() == 'basic':
                demo_basic_conversation()
            elif user_input.lower() == 'search':
                demo_search_and_analysis()
            elif user_input.lower() == 'decision':
                demo_decision_support()
            elif user_input.lower() == 'project':
                demo_project_management()
            elif user_input.lower() == 'performance':
                demo_performance_test()
            elif user_input == '':
                continue
            else:
                # 正常对话
                print(f"\n用户输入: {user_input}")
                response = bot.chat(user_input)
                print(f"机器人响应:\n{response}")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("请重试或联系技术支持。")


if __name__ == "__main__":
    print("请选择运行模式:")
    print("1. 交互式演示 (推荐)")
    print("2. 运行所有演示")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        run_all_demos()
    else:
        interactive_demo()
