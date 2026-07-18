# CKM 死亡风险预测系统

基于 Cox 比例风险模型的个体化生存预测工具。

## 功能

- 输入 11 项临床指标预测 3/5/8 年生存概率
- 生成个体化生存曲线
- 与基准人群对比分析
- 识别主要风险因素

## 模型性能

- 数据来源: CHARLS 队列 (N=6,953)
- 模型: Cox Proportional Hazards (15-variable, z-scored)
- 5-Fold CV C-index: 0.805
- Train C-index: 0.808
- Test C-index: 0.785
- Test Brier Score: 0.057
- 特征: 11 项 (年龄、BMI、胱抑素C、C反应蛋白、肌酐/胱抑素C比值、性别、教育、婚姻、高血压、肺疾病、CKM分期)

## 技术栈

- Streamlit + Python
- Plotly 交互式可视化
- Cox PH 回归模型 (z-score 标准化)

## 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 在线访问

部署于 Streamlit Cloud: https://82pjftz5hpnnr2hsgzabtu.streamlit.app/
