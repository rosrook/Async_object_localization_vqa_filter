"""
VQA评估运行脚本 - 完整示例
适配你的数据格式：包含image_base64字段
"""

import sys
from data_loader import VQADataLoader
from vqa_evaluator import VQAEvaluator


def main():
    print("=" * 70)
    print(" " * 20 + "VQA模型评估系统")
    print("=" * 70)
    
    # ============ 配置区域 - 修改这里 ============
    CONFIG = {
        # 模型配置
        # 使用与 vlmevalkit 相同的完整 LLaVA-OneVision 模型目录（而不是 hf_stage2 增量权重）
        'model_path': "/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Instruct",
        'device': "cuda",                    # "cuda" 或 "cpu"
        
        # 数据配置
        'data_file': "vqa.json",            # 你的数据文件
        'include_options': True,             # True: 使用full_question, False: 使用question
        
        # 评估配置
        'max_samples': None,                 # None: 全部, 或数字(如10)用于快速测试
        'strict_match': False,               # False: 宽松匹配, True: 严格匹配
        
        # 输出配置
        'output_file': "evaluation_results.json",
        'save_sample_images': True,          # 是否保存样本图片检查
        'num_sample_images': 3,              # 保存几张样本图片
    }
    # ============================================
    
    try:
        # 步骤1: 加载数据
        print("\n" + "=" * 70)
        print("步骤 1/4: 加载和预处理数据")
        print("=" * 70)
        
        loader = VQADataLoader(CONFIG['data_file'])
        loader.print_statistics()
        
        # 处理数据
        processed_data = loader.process_all(
            include_options=CONFIG['include_options'],
            max_samples=CONFIG['max_samples']
        )
        
        if not processed_data:
            print("错误: 没有成功处理任何数据！")
            return
        
        # 可选: 保存样本图片
        if CONFIG['save_sample_images']:
            print(f"\n保存前{CONFIG['num_sample_images']}张样本图片用于检查...")
            loader.save_sample_images(n=CONFIG['num_sample_images'])
        
        # 步骤2: 加载模型
        print("\n" + "=" * 70)
        print("步骤 2/4: 加载VQA模型")
        print("=" * 70)
        
        evaluator = VQAEvaluator(
            model_path=CONFIG['model_path'],
            device=CONFIG['device']
        )
        
        # 步骤3: 运行评估
        print("\n" + "=" * 70)
        print("步骤 3/4: 运行模型评估")
        print("=" * 70)
        print(f"评估样本数: {len(processed_data)}")
        print(f"答案匹配模式: {'严格匹配' if CONFIG['strict_match'] else '宽松匹配'}")
        
        results = evaluator.evaluate_dataset(
            processed_data=processed_data,
            output_file=CONFIG['output_file'],
            strict_match=CONFIG['strict_match']
        )
        
        # 步骤4: 显示结果摘要
        print("\n" + "=" * 70)
        print("步骤 4/4: 评估结果摘要")
        print("=" * 70)
        
        total = results['total_samples']
        correct = results['correct_predictions']
        accuracy = results['accuracy']
        
        print(f"\n总样本数: {total}")
        print(f"正确预测: {correct}")
        print(f"错误预测: {total - correct}")
        print(f"准确率: {accuracy:.2%}")
        
        # 难度评估
        print("\n数据集难度评估:")
        if accuracy > 0.95:
            difficulty = "过于简单"
            suggestion = "建议增加难度，添加更复杂的问题"
        elif accuracy > 0.85:
            difficulty = "适中偏易"
            suggestion = "可以适当增加难度"
        elif accuracy > 0.70:
            difficulty = "难度适中"
            suggestion = "适合当前模型训练"
        elif accuracy > 0.50:
            difficulty = "较难"
            suggestion = "适合挑战模型能力上限"
        else:
            difficulty = "非常难"
            suggestion = "可能需要更强的模型或更多训练数据"
        
        print(f"  难度级别: {difficulty}")
        print(f"  建议: {suggestion}")
        
        # 显示几个样本结果
        print("\n" + "-" * 70)
        print("样本结果示例（前3个）:")
        print("-" * 70)
        
        for i, r in enumerate(results['results'][:3]):
            status = "✓ 正确" if r['correct'] else "✗ 错误"
            print(f"\n样本 {i+1} (ID: {r['id']}) - {status}")
            print(f"  问题: {r['question'][:100]}{'...' if len(r['question']) > 100 else ''}")
            print(f"  正确答案: {r['ground_truth']}")
            print(f"  模型预测: {r['prediction'][:100]}{'...' if len(r['prediction']) > 100 else ''}")
        
        print("\n" + "=" * 70)
        print("评估完成！")
        print("=" * 70)
        print(f"\n详细结果已保存到: {CONFIG['output_file']}")
        print("运行以下命令查看详细分析:")
        print("  python vqa_analyzer.py")
        print("\n或者在Python中:")
        print("  from vqa_analyzer import VQAAnalyzer")
        print(f"  analyzer = VQAAnalyzer('{CONFIG['output_file']}')")
        print("  analyzer.print_summary()")
        print("  analyzer.show_error_cases()")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {e}")
        print("请检查配置中的文件路径是否正确")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()