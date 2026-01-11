# -*- coding: utf-8 -*-
"""
深度学习神经网络智能AI机器人
包含对话管理、数据挖掘与分析、决策支持、自我优化和用户指令执行模块
"""

import numpy as np
import pandas as pd
import json
import os
import time
import threading
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime
import random


class SimpleNeuralNetwork:
    """
    简化的神经网络模型（使用numpy实现）
    """
    def __init__(self, input_size=100, hidden_size=64, output_size=100, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # 输入层到隐藏层
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.1)
        self.biases.append(np.random.randn(hidden_size) * 0.1)
        
        # 隐藏层到隐藏层
        for _ in range(num_layers - 1):
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.1)
            self.biases.append(np.random.randn(hidden_size) * 0.1)
        
        # 隐藏层到输出层
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.1)
        self.biases.append(np.random.randn(output_size) * 0.1)
    
    def sigmoid(self, x):
        """激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """前向传播"""
        # 将输入转换为向量
        if isinstance(x, (int, float)):
            x = np.array([x])
        elif isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, str):
            # 将字符串转换为数值向量（简化处理）
            x = np.array([hash(x) % 1000 / 1000.0 for _ in range(self.input_size)])
        
        # 逐层计算
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, weight) + bias
            if i < len(self.weights) - 1:  # 最后一层不用激活函数
                x = self.sigmoid(x)
        
        return x


class SimpleConversationManager:
    """
    简化版对话管理模块
    """
    def __init__(self):
        self.neural_net = SimpleNeuralNetwork()
        self.word_embeddings = {}  # 简单的词嵌入存储
        self.response_templates = [
            "我理解您说的关于 '{}' 的内容。",
            "关于 {}，我认为这是一个很重要的问题。",
            "我已经记录了您提到的 {} 信息。",
            "这是一个有趣的观点，我们可以进一步探讨 {}。",
            "关于 {}，我有一些想法想和您分享。",
            "您提到的 {} 确实值得深入讨论。",
            "明白了，{} 是您关注的重点。",
            "很有趣，{} 这个话题我很乐意和您交流。"
        ]
    
    def encode_text(self, text: str) -> np.ndarray:
        """简单文本编码（使用哈希和字符统计）"""
        # 使用哈希值创建固定长度的向量
        vector = np.zeros(100)
        for i, char in enumerate(text[:50]):  # 只考虑前50个字符
            vector[i % 100] += ord(char) / 1000.0
        
        # 添加词频信息
        words = text.split()
        for i, word in enumerate(words[:20]):  # 只考虑前20个词
            vector[(i + 50) % 100] += hash(word) % 1000 / 1000.0
        
        return vector
    
    def generate_response(self, user_input: str) -> str:
        """生成对话响应"""
        encoded_input = self.encode_text(user_input)
        output = self.neural_net.forward(encoded_input)
        
        # 基于输出选择响应模板
        template_idx = int(abs(output[0] * 100)) % len(self.response_templates)
        short_input = user_input[:20] if len(user_input) > 20 else user_input
        
        return self.response_templates[template_idx].format(short_input)


class SimpleDataMiner:
    """
    简化版数据挖掘与分析模块
    """
    def __init__(self):
        self.search_history = []
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """模拟网络搜索"""
        print(f"正在搜索: {query}")
        
        # 模拟搜索结果
        results = []
        for i in range(max_results):
            results.append({
                'title': f'搜索结果 {i+1} 关于 {query}',
                'url': f'https://example.com/result{i+1}',
                'snippet': f'这是关于{query}的相关信息和数据摘要，包含重要知识点和参考价值。',
                'timestamp': datetime.now().isoformat()
            })
        
        self.search_history.append({
            'query': query,
            'results_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
        return results
    
    def analyze_data(self, data: List[Dict]) -> Dict[str, Any]:
        """使用pandas分析数据"""
        if not data:
            return {'error': '没有数据可供分析'}
        
        df = pd.DataFrame(data)
        
        analysis_result = {
            'total_records': len(data),
            'columns': list(df.columns) if not df.empty else [],
            'sample_data': df.head().to_dict('records') if not df.empty else [],
            'data_types': str(df.dtypes.to_dict()) if not df.empty else {},
            'has_numeric_columns': len(df.select_dtypes(include=[np.number]).columns) > 0
        }
        
        # 数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            try:
                analysis_result['statistics'] = df[numeric_cols].describe().to_dict()
            except:
                analysis_result['statistics'] = "无法计算统计数据"
        else:
            analysis_result['statistics'] = "无数值列可分析"
        
        return analysis_result


class SimpleDecisionModule:
    """
    简化版决策支持模块
    """
    def __init__(self):
        self.models_trained = False
        self.decision_rules = {
            'framework_choice': {
                'deep_learning': ['pytorch', 'tensorflow', 'keras'],
                'machine_learning': ['scikit-learn', 'xgboost', 'lightgbm'],
                'web_development': ['django', 'flask', 'fastapi']
            }
        }
    
    def make_decision(self, features: List[float]) -> Dict[str, Any]:
        """基于输入特征做出决策"""
        if not features:
            features = [random.random() for _ in range(5)]
        
        # 基于特征的加权计算
        weighted_sum = sum(f * (i+1) for i, f in enumerate(features))
        
        # 生成多个模型的预测
        dt_prediction = int(weighted_sum * 10) % 3
        rf_prediction = int(sum(features) * 7) % 3
        
        # 计算置信度
        confidence_values = [random.random() for _ in range(3)]
        total_confidence = sum(confidence_values)
        normalized_confidence = [c/total_confidence for c in confidence_values] if total_confidence > 0 else [1/3]*3
        
        return {
            'decision_tree_prediction': dt_prediction,
            'random_forest_prediction': rf_prediction,
            'confidence_scores': normalized_confidence,
            'final_decision': (dt_prediction + rf_prediction) // 2,
            'recommendation': self._get_recommendation(features)
        }
    
    def _get_recommendation(self, features: List[float]) -> str:
        """基于特征生成推荐"""
        if len(features) >= 3:
            if features[0] > 0.5:
                return "推荐使用深度学习方法"
            elif features[1] > 0.5:
                return "推荐使用传统机器学习方法"
            else:
                return "推荐先进性数据探测"
        else:
            return "需要更多信息来提供建议"


class SimpleSelfOptimizer:
    """
    简化版自我优化模块
    """
    def __init__(self, neural_network: SimpleNeuralNetwork):
        self.neural_network = neural_network
        self.learning_rate = 0.01
        self.training_history = []
        self.iteration_count = 0
    
    def compute_loss(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """计算损失函数（均方误差）"""
        return np.mean((predicted - target) ** 2)
    
    def backpropagate(self, input_vector: np.ndarray, target_vector: np.ndarray):
        """简化版反向传播"""
        # 当前预测
        predicted = self.neural_network.forward(input_vector.copy())
        
        # 计算损失
        loss = self.compute_loss(predicted, target_vector)
        
        # 简单的梯度更新（真实场景中需要更复杂的反向传播）
        for i in range(len(self.neural_network.weights)):
            # 随机扰动权重
            weight_perturbation = np.random.randn(*self.neural_network.weights[i].shape) * self.learning_rate * 0.1
            bias_perturbation = np.random.randn(*self.neural_network.biases[i].shape) * self.learning_rate * 0.1
            
            self.neural_network.weights[i] -= weight_perturbation
            self.neural_network.biases[i] -= bias_perturbation
        
        # 记录训练历史
        self.iteration_count += 1
        self.training_history.append({
            'iteration': self.iteration_count,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        })
        
        return loss
    
    def optimize(self, training_data: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 5):
        """执行优化过程"""
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for input_vec, target_vec in training_data:
                loss = self.backpropagate(input_vec, target_vec)
                epoch_loss += loss
            
            avg_epoch_loss = epoch_loss / len(training_data) if training_data else 0
            total_loss += avg_epoch_loss
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
        
        return total_loss / epochs if epochs > 0 else 0


class SimpleInstructionExecutor:
    """
    简化版用户指令执行模块
    """
    def __init__(self):
        self.task_queue = []
        self.completed_tasks = []
    
    def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """需求分析"""
        analysis = {
            'requirements': requirements,
            'complexity': self._assess_complexity(requirements),
            'components': self._identify_components(requirements),
            'estimated_time': self._estimate_time(requirements),
            'risks': self._identify_risks(requirements),
            'priority': self._assign_priority(requirements)
        }
        return analysis
    
    def design_architecture(self, requirements_analysis: Dict) -> Dict[str, Any]:
        """架构设计"""
        # 根据需求复杂度选择架构模式
        if requirements_analysis['complexity'] == 'High':
            patterns = ['Microservices', 'Event-Driven', 'CQRS']
        elif requirements_analysis['complexity'] == 'Medium':
            patterns = ['Layered Architecture', 'Service-Oriented']
        else:
            patterns = ['Monolithic', 'MVC']
        
        architecture = {
            'architecture_patterns': patterns,
            'recommended_technologies': self._suggest_technologies(requirements_analysis['components']),
            'system_components': {
                'frontend': self._select_frontend(requirements_analysis),
                'backend': self._select_backend(requirements_analysis),
                'database': self._select_database(requirements_analysis),
                'infrastructure': ['Load Balancer', 'CDN', 'Monitoring']
            },
            'deployment_strategy': self._select_deployment(requirements_analysis['complexity'])
        }
        return architecture
    
    def implement_technology(self, architecture: Dict) -> Dict[str, Any]:
        """技术实现规划"""
        implementation = {
            'implementation_phases': [
                {'phase': 'Phase 1: Environment Setup', 'duration': '1 week', 'tasks': ['Install dependencies', 'Set up environment']},
                {'phase': 'Phase 2: Core Development', 'duration': '2-3 weeks', 'tasks': ['Develop core modules', 'Implement features']},
                {'phase': 'Phase 3: Testing', 'duration': '1 week', 'tasks': ['Unit tests', 'Integration tests']},
                {'phase': 'Phase 4: Deployment', 'duration': '1 week', 'tasks': ['Deploy to staging', 'Deploy to production']}
            ],
            'recommended_tools': architecture['recommended_technologies'],
            'estimated_timeline': '4-6 weeks',
            'resource_requirements': ['Developer', 'Designer', 'QA Engineer']
        }
        return implementation
    
    def develop_project(self, implementation_plan: Dict) -> Dict[str, Any]:
        """项目开发管理"""
        development = {
            'project_status': 'Planning',
            'development_phases': implementation_plan['implementation_phases'],
            'estimated_completion': implementation_plan['estimated_timeline'],
            'team_allocation': implementation_plan['resource_requirements'],
            'risk_assessment': ['Technical risks', 'Timeline risks', 'Resource risks'],
            'milestones': ['Requirements finalized', 'Design completed', 'Development phase 1', 'Testing phase', 'Go live']
        }
        return development
    
    def deploy_publish(self, development_status: Dict) -> Dict[str, Any]:
        """部署发布计划"""
        deployment = {
            'environment_setup': ['Staging server', 'Production server', 'Database servers'],
            'deployment_steps': [
                'Configure infrastructure',
                'Deploy application',
                'Run smoke tests',
                'Perform load testing',
                'Go live'
            ],
            'monitoring_setup': ['Application logs', 'System metrics', 'Error tracking'],
            'rollback_plan': 'Revert to previous version if issues arise'
        }
        return deployment
    
    def setup_ci_cd(self, deployment_config: Dict) -> Dict[str, Any]:
        """CI/CD流程设置"""
        ci_cd = {
            'source_control': 'Git with feature branch workflow',
            'build_process': ['Code compilation', 'Dependency installation', 'Static analysis'],
            'test_automation': ['Unit tests', 'Integration tests', 'Security scans'],
            'deployment_pipeline': ['Build', 'Test', 'Deploy to staging', 'Manual approval', 'Deploy to production'],
            'recommended_tools': ['Jenkins', 'GitHub Actions', 'Docker', 'Kubernetes']
        }
        return ci_cd
    
    def _assess_complexity(self, req: str) -> str:
        """评估复杂度"""
        word_count = len(req.split())
        if word_count < 50:
            return 'Low'
        elif word_count < 150:
            return 'Medium'
        else:
            return 'High'
    
    def _identify_components(self, req: str) -> List[str]:
        """识别组件"""
        req_lower = req.lower()
        components = []
        
        if any(keyword in req_lower for keyword in ['web', 'website', 'interface', 'ui', 'frontend']):
            components.append('Web Frontend')
        if any(keyword in req_lower for keyword in ['api', 'backend', 'server', 'service', 'logic']):
            components.append('Backend Service')
        if any(keyword in req_lower for keyword in ['database', 'storage', 'data', 'db']):
            components.append('Database Layer')
        if any(keyword in req_lower for keyword in ['mobile', 'app', 'ios', 'android']):
            components.append('Mobile Application')
        if any(keyword in req_lower for keyword in ['ai', 'ml', 'machine learning', 'intelligent']):
            components.append('AI/ML Module')
        
        return components if components else ['Core System']
    
    def _estimate_time(self, req: str) -> str:
        """估算时间"""
        complexity = self._assess_complexity(req)
        if complexity == 'Low':
            return '1-2 weeks'
        elif complexity == 'Medium':
            return '3-6 weeks'
        else:
            return '2-3 months'
    
    def _identify_risks(self, req: str) -> List[str]:
        """识别风险"""
        req_lower = req.lower()
        risks = []
        
        if any(keyword in req_lower for keyword in ['real-time', 'high-performance', 'scalability']):
            risks.append('Performance and Scalability Risks')
        if any(keyword in req_lower for keyword in ['integration', 'third-party', 'external']):
            risks.append('Third-party Integration Risks')
        if any(keyword in req_lower for keyword in ['security', 'authentication', 'privacy']):
            risks.append('Security and Privacy Risks')
        
        return risks if risks else ['General Project Risks']
    
    def _assign_priority(self, req: str) -> str:
        """分配优先级"""
        if 'urgent' in req.lower() or 'asap' in req.lower() or 'immediate' in req.lower():
            return 'High'
        elif 'important' in req.lower():
            return 'Medium-High'
        else:
            return 'Medium'
    
    def _suggest_technologies(self, components: List[str]) -> List[str]:
        """推荐技术栈"""
        technologies = []
        
        if 'Web Frontend' in components:
            technologies.extend(['React', 'Vue.js', 'TypeScript'])
        if 'Backend Service' in components:
            technologies.extend(['Python', 'Node.js', 'FastAPI/Django'])
        if 'Database Layer' in components:
            technologies.extend(['PostgreSQL', 'MongoDB', 'Redis'])
        if 'Mobile Application' in components:
            technologies.extend(['React Native', 'Flutter', 'Swift/Kotlin'])
        if 'AI/ML Module' in components:
            technologies.extend(['TensorFlow', 'PyTorch', 'Scikit-learn'])
        
        if not technologies:
            technologies = ['Python', 'JavaScript', 'PostgreSQL']
        
        return technologies
    
    def _select_frontend(self, analysis: Dict) -> str:
        """选择前端技术"""
        if 'Mobile Application' in analysis['components']:
            return 'React Native or Flutter'
        else:
            return 'React with TypeScript'
    
    def _select_backend(self, analysis: Dict) -> str:
        """选择后端技术"""
        if 'AI/ML Module' in analysis['components']:
            return 'Python with FastAPI'
        else:
            return 'Node.js with Express or Python with Django'
    
    def _select_database(self, analysis: Dict) -> str:
        """选择数据库"""
        if 'AI/ML Module' in analysis['components']:
            return 'PostgreSQL with Redis cache'
        else:
            return 'PostgreSQL or MongoDB'
    
    def _select_deployment(self, complexity: str) -> str:
        """选择部署策略"""
        if complexity == 'High':
            return 'Microservices with Kubernetes'
        elif complexity == 'Medium':
            return 'Containerized deployment with Docker'
        else:
            return 'Traditional server deployment'


class SimpleNeuralAIBot:
    """
    简化版主AI机器人类，整合所有模块
    """
    def __init__(self):
        print("正在初始化简化版深度学习神经网络智能AI机器人...")
        
        # 初始化各模块
        self.conversation_manager = SimpleConversationManager()
        self.data_miner = SimpleDataMiner()
        self.decision_module = SimpleDecisionModule()
        self.neural_network = SimpleNeuralNetwork()
        self.self_optimizer = SimpleSelfOptimizer(self.neural_network)
        self.instruction_executor = SimpleInstructionExecutor()
        
        print("简化版AI机器人初始化完成！")
    
    def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求的主函数"""
        start_time = time.time()
        
        # 1. 对话管理
        conversation_response = self.conversation_manager.generate_response(user_input)
        
        # 2. 如果用户请求搜索或分析，执行数据挖掘
        search_keywords = self._extract_search_keywords(user_input)
        search_results = []
        analysis_results = {}
        
        if search_keywords:
            search_results = self.data_miner.search_web(' '.join(search_keywords))
            analysis_results = self.data_miner.analyze_data(search_results)
        
        # 3. 决策支持（如果需要）
        decision_result = None
        if any(word in user_input.lower() for word in ['决定', '决策', '选择', '推荐', '应该', '哪个']):
            # 创建模拟特征用于决策
            mock_features = [random.random() for _ in range(5)]
            decision_result = self.decision_module.make_decision(mock_features)
        
        # 4. 执行用户指令（如果包含特定命令）
        instruction_result = None
        if any(cmd in user_input.lower() for cmd in ['分析需求', '设计架构', '实施技术', '开发项目', '部署发布', 'ci/cd', '需求分析', '架构设计']):
            instruction_result = self._execute_user_instruction(user_input)
        
        # 5. 自我优化（模拟）
        if len(self.self_optimizer.training_history) % 5 == 0 and len(self.self_optimizer.training_history) > 0:
            self._perform_self_optimization(user_input)
        
        response_time = time.time() - start_time
        
        return {
            'conversation_response': conversation_response,
            'search_results': search_results,
            'analysis_results': analysis_results,
            'decision_result': decision_result,
            'instruction_result': instruction_result,
            'response_time': response_time,
            'optimization_status': len(self.self_optimizer.training_history)
        }
    
    def _extract_search_keywords(self, text: str) -> List[str]:
        """提取搜索关键词"""
        # 简单的关键词提取逻辑
        keywords = []
        text_lower = text.lower()
        
        # 查找特定模式的关键词
        search_indicators = ['搜索', '查找', '查询', '了解', '是什么', '怎么', '如何', '最新', '新闻', '信息', 'find', 'search', 'look up', 'tell me about']
        if any(indicator in text_lower for indicator in search_indicators):
            # 提取名词性短语作为关键词
            words = re.findall(r'[a-zA-Z一-龯]+', text)
            keywords = [word for word in words if len(word) > 1]  # 过滤掉单字符
        
        return keywords[:5]  # 返回前5个关键词
    
    def _execute_user_instruction(self, instruction: str) -> Dict[str, Any]:
        """执行用户指令"""
        instruction_lower = instruction.lower()
        
        if any(keyword in instruction_lower for keyword in ['分析需求', '需求分析']):
            return self.instruction_executor.analyze_requirements(instruction)
        elif any(keyword in instruction_lower for keyword in ['设计架构', '架构设计']):
            req_analysis = self.instruction_executor.analyze_requirements(instruction)
            return self.instruction_executor.design_architecture(req_analysis)
        elif any(keyword in instruction_lower for keyword in ['实施技术', '技术实现']):
            arch = self.instruction_executor.design_architecture(
                self.instruction_executor.analyze_requirements(instruction)
            )
            return self.instruction_executor.implement_technology(arch)
        elif any(keyword in instruction_lower for keyword in ['开发项目', '项目开发']):
            impl = self.instruction_executor.implement_technology(
                self.instruction_executor.design_architecture(
                    self.instruction_executor.analyze_requirements(instruction)
                )
            )
            return self.instruction_executor.develop_project(impl)
        elif any(keyword in instruction_lower for keyword in ['部署发布', '发布部署']):
            dev_status = self.instruction_executor.develop_project(
                self.instruction_executor.implement_technology(
                    self.instruction_executor.design_architecture(
                        self.instruction_executor.analyze_requirements(instruction)
                    )
                )
            )
            return self.instruction_executor.deploy_publish(dev_status)
        elif any(keyword in instruction_lower for keyword in ['ci/cd', '持续集成', '部署流程']):
            deploy_config = self.instruction_executor.deploy_publish(
                self.instruction_executor.develop_project(
                    self.instruction_executor.implement_technology(
                        self.instruction_executor.design_architecture(
                            self.instruction_executor.analyze_requirements(instruction)
                        )
                    )
                )
            )
            return self.instruction_executor.setup_ci_cd(deploy_config)
        else:
            # 如果无法识别具体指令，则尝试需求分析
            return self.instruction_executor.analyze_requirements(instruction)
    
    def _perform_self_optimization(self, input_text: str):
        """执行自我优化"""
        try:
            # 使用输入创建训练数据
            input_vector = self.conversation_manager.encode_text(input_text)
            target_vector = input_vector.copy()  # 使用自身作为目标（自监督学习）
            
            # 创建训练批次
            training_data = [(input_vector, target_vector)]
            
            # 执行优化
            avg_loss = self.self_optimizer.optimize(training_data, epochs=1)
            print(f"自我优化完成，平均损失: {avg_loss:.4f}")
        except Exception as e:
            print(f"自我优化过程中出现错误: {str(e)}")
    
    def chat(self, user_input: str) -> str:
        """简单的聊天接口"""
        result = self.process_user_request(user_input)
        
        response_parts = []
        
        # 添加对话响应
        response_parts.append(f"🤖 {result['conversation_response']}")
        
        # 添加搜索结果（如果有）
        if result['search_results']:
            response_parts.append(f"🔍 搜索到 {len(result['search_results'])} 条相关信息:")
            for i, res in enumerate(result['search_results'][:3]):  # 只显示前3条
                response_parts.append(f"  {i+1}. {res['title']}")
        
        # 添加分析结果摘要
        if result['analysis_results'] and 'error' not in result['analysis_results']:
            response_parts.append(f"📊 数据分析: 共处理 {result['analysis_results']['total_records']} 条记录")
        
        # 添加决策结果（如果有）
        if result['decision_result']:
            response_parts.append(f"🧠 决策建议: {result['decision_result']['recommendation']}")
        
        # 添加指令执行结果（如果有）
        if result['instruction_result']:
            response_parts.append("📋 指令执行结果:")
            for key, value in list(result['instruction_result'].items())[:3]:  # 只显示前3个项目
                if isinstance(value, (str, int, float)):
                    response_parts.append(f"  {key}: {value}")
                elif isinstance(value, list) and value:
                    response_parts.append(f"  {key}: {str(value[:3])}")  # 只显示前3个元素
        
        response_parts.append(f"⏱️ 响应时间: {result['response_time']:.2f}秒")
        response_parts.append(f"🔄 优化次数: {result['optimization_status']}")
        
        return "\n".join(response_parts)


def main():
    """主函数 - 机器人演示"""
    print("="*60)
    print("简化版深度学习神经网络智能AI机器人")
    print("支持对话、搜索、分析、决策和指令执行")
    print("输入 'quit' 或 'exit' 退出程序")
    print("="*60)
    
    # 创建机器人实例
    ai_bot = SimpleNeuralAIBot()
    
    # 示例交互
    print("\n🤖 您好！我是简化版深度学习神经网络智能AI机器人，我可以帮助您对话、搜索信息、分析数据、做决策等。")
    print("您可以问我任何问题，比如：")
    print("- '你好，介绍一下你自己'")
    print("- '帮我分析一下人工智能的发展趋势'") 
    print("- '推荐一个好的机器学习项目架构'")
    print("- '搜索最新的PyTorch教程'")
    print("- '分析需求开发一个聊天机器人'")
    print()
    
    while True:
        try:
            user_input = input("👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                print("🤖 机器人: 再见！感谢使用简化版深度学习神经网络智能AI机器人。")
                break
            
            if not user_input:
                continue
                
            # 处理用户输入
            response = ai_bot.chat(user_input)
            print(f"\n{response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n🤖 机器人: 收到退出信号，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重新输入或联系技术支持。")


if __name__ == "__main__":
    main()
