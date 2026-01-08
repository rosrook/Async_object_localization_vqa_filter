"""
VQA评估结果分析器
提供数据分析和可视化功能
"""

import json
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Any

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class VQAAnalyzer:
    """VQA结果分析器"""
    
    def __init__(self, results_file: str):
        """
        初始化分析器
        
        Args:
            results_file: 评估结果JSON文件路径
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def print_summary(self):
        """打印评估摘要"""
        print("=" * 60)
        print("VQA评估结果摘要")
        print("=" * 60)
        print(f"模型: {self.data.get('model_path', 'N/A')}")
        print(f"设备: {self.data.get('device', 'N/A')}")
        print(f"匹配模式: {'严格' if self.data.get('strict_match') else '宽松'}")
        print(f"总样本数: {self.data['total_samples']}")
        print(f"正确预测数: {self.data['correct_predictions']}")
        print(f"错误预测数: {self.data['total_samples'] - self.data['correct_predictions']}")
        print(f"准确率: {self.data['accuracy']:.2%}")
        print("=" * 60)
    
    def show_error_cases(self, n: int = 10):
        """
        显示错误样本
        
        Args:
            n: 显示的错误样本数量
        """
        errors = [r for r in self.data['results'] if not r['correct']]
        
        print(f"\n错误样本分析 (共 {len(errors)} 个错误):")
        print("-" * 60)
        
        if not errors:
            print("恭喜！没有错误样本！")
            return
        
        for i, err in enumerate(errors[:n]):
            print(f"\n错误 #{i+1} (ID: {err.get('id', 'N/A')}):")
            print(f"  问题: {err['question'][:100]}{'...' if len(err['question']) > 100 else ''}")
            print(f"  正确答案: {err['ground_truth']}")
            print(f"  模型预测: {err['prediction'][:150]}{'...' if len(err['prediction']) > 150 else ''}")
            
            # 如果有元数据，显示问题类型
            if 'metadata' in err and err['metadata'].get('question_type'):
                print(f"  问题类型: {err['metadata']['question_type']}")
    
    def analyze_question_types(self):
        """分析不同问题类型的准确率"""
        # 检查是否有元数据中的question_type
        has_metadata = any('metadata' in r and r['metadata'].get('question_type') 
                          for r in self.data['results'])
        
        if has_metadata:
            # 使用元数据中的问题类型
            type_stats = {}
            for result in self.data['results']:
                if 'metadata' in result and result['metadata'].get('question_type'):
                    qtype = result['metadata']['question_type']
                else:
                    qtype = '未分类'
                
                if qtype not in type_stats:
                    type_stats[qtype] = {'correct': 0, 'total': 0}
                
                type_stats[qtype]['total'] += 1
                if result['correct']:
                    type_stats[qtype]['correct'] += 1
        else:
            # 基于关键词的简单分类
            question_types = {
                '计数': ['多少', '几个', 'how many', 'count'],
                '颜色': ['颜色', '什么色', 'color', 'what color'],
                '物体识别': ['是什么', '什么东西', 'what is', 'what are', 'identify'],
                '位置': ['在哪', '位置', 'where', 'location'],
                '动作': ['在做什么', '正在', 'doing', 'what doing'],
                '是否判断': ['是不是', '有没有', 'is there', 'are there'],
                '选择题': ['which', 'select', 'choose', 'option', '选项']
            }
            
            type_stats = {qtype: {'correct': 0, 'total': 0} for qtype in question_types.keys()}
            type_stats['其他'] = {'correct': 0, 'total': 0}
            
            for result in self.data['results']:
                question = result['question'].lower()
                matched = False
                
                for qtype, keywords in question_types.items():
                    if any(kw in question for kw in keywords):
                        type_stats[qtype]['total'] += 1
                        if result['correct']:
                            type_stats[qtype]['correct'] += 1
                        matched = True
                        break
                
                if not matched:
                    type_stats['其他']['total'] += 1
                    if result['correct']:
                        type_stats['其他']['correct'] += 1
        
        print("\n问题类型准确率分析:")
        print("-" * 60)
        for qtype, stats in sorted(type_stats.items(), key=lambda x: -x[1]['total']):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"{qtype:15s}: {acc:6.2%}  ({stats['correct']}/{stats['total']})")
        
        return type_stats
    
    def plot_accuracy_distribution(self, save_path: str = "accuracy_plot.png"):
        """绘制准确率分布图"""
        type_stats = self.analyze_question_types()
        
        # 过滤掉总数为0的类型
        filtered_stats = {k: v for k, v in type_stats.items() if v['total'] > 0}
        
        if not filtered_stats:
            print("没有足够的数据生成图表")
            return
        
        # 按样本数排序
        sorted_items = sorted(filtered_stats.items(), key=lambda x: -x[1]['total'])
        types = [item[0] for item in sorted_items]
        accuracies = [item[1]['correct']/item[1]['total'] if item[1]['total'] > 0 else 0 
                     for item in sorted_items]
        totals = [item[1]['total'] for item in sorted_items]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(types)), accuracies, color='skyblue', edgecolor='navy')
        
        # 为不同准确率范围使用不同颜色
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc >= 0.9:
                bar.set_color('lightgreen')
            elif acc >= 0.7:
                bar.set_color('skyblue')
            elif acc >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        plt.xlabel('问题类型', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.title('不同问题类型的准确率分布', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.xticks(range(len(types)), types, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上标注数值和样本数
        for i, (acc, total) in enumerate(zip(accuracies, totals)):
            plt.text(i, acc + 0.02, f'{acc:.1%}\n(n={total})', 
                    ha='center', va='bottom', fontsize=9)
        
        # 添加整体准确率线
        overall_acc = self.data['accuracy']
        plt.axhline(y=overall_acc, color='red', linestyle='--', 
                   label=f'整体准确率: {overall_acc:.1%}', linewidth=2)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存到: {save_path}")
        plt.close()
    
    def export_error_report(self, output_file: str = "error_report.txt"):
        """导出错误报告"""
        errors = [r for r in self.data['results'] if not r['correct']]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VQA错误样本详细报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"模型: {self.data.get('model_path', 'N/A')}\n")
            f.write(f"总样本数: {self.data['total_samples']}\n")
            f.write(f"总错误数: {len(errors)}\n")
            f.write(f"错误率: {1 - self.data['accuracy']:.2%}\n")
            f.write(f"准确率: {self.data['accuracy']:.2%}\n")
            f.write("=" * 80 + "\n\n")
            
            if not errors:
                f.write("恭喜！没有错误样本！\n")
            else:
                for i, err in enumerate(errors, 1):
                    f.write(f"错误 #{i}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"ID: {err.get('id', 'N/A')}\n")
                    f.write(f"问题: {err['question']}\n")
                    f.write(f"正确答案: {', '.join(err['ground_truth'])}\n")
                    f.write(f"模型预测: {err['prediction']}\n")
                    
                    if 'metadata' in err:
                        metadata = err['metadata']
                        if metadata.get('question_type'):
                            f.write(f"问题类型: {metadata['question_type']}\n")
                        if metadata.get('answer_type'):
                            f.write(f"答案类型: {metadata['answer_type']}\n")
                    
                    f.write("\n")
        
        print(f"错误报告已导出到: {output_file}")
    
    def get_difficulty_assessment(self) -> Dict[str, str]:
        """获取数据集难度评估"""
        accuracy = self.data['accuracy']
        
        if accuracy > 0.95:
            return {
                'level': '过于简单',
                'description': '模型表现优秀，数据集可能过于简单',
                'suggestion': '建议增加难度，添加更复杂的问题或更具挑战性的图片'
            }
        elif accuracy > 0.85:
            return {
                'level': '适中偏易',
                'description': '模型表现良好',
                'suggestion': '可以适当增加难度，或保持当前难度进行训练'
            }
        elif accuracy > 0.70:
            return {
                'level': '难度适中',
                'description': '难度适合当前模型',
                'suggestion': '当前难度适合用于模型训练和评估'
            }
        elif accuracy > 0.50:
            return {
                'level': '较难',
                'description': '对模型有一定挑战',
                'suggestion': '适合挑战模型能力上限，或考虑增加训练数据'
            }
        else:
            return {
                'level': '非常难',
                'description': '模型表现较弱',
                'suggestion': '可能需要更强的模型、更多训练数据或降低数据难度'
            }


def main():
    """使用示例"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "evaluation_results.json"
    
    print(f"正在分析: {results_file}\n")
    
    try:
        # 创建分析器
        analyzer = VQAAnalyzer(results_file)
        
        # 打印摘要
        analyzer.print_summary()
        
        # 难度评估
        difficulty = analyzer.get_difficulty_assessment()
        print(f"\n数据集难度评估:")
        print(f"  级别: {difficulty['level']}")
        print(f"  描述: {difficulty['description']}")
        print(f"  建议: {difficulty['suggestion']}")
        
        # 显示错误案例
        analyzer.show_error_cases(n=5)
        
        # 分析问题类型
        analyzer.analyze_question_types()
        
        # 绘制图表
        analyzer.plot_accuracy_distribution()
        
        # 导出错误报告
        analyzer.export_error_report()
        
        print("\n分析完成！")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {results_file}")
        print("请确保已运行评估并生成了结果文件")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()