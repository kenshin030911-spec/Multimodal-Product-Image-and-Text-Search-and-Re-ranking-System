# outputs

这里存放训练报告、数据检查报告、离线评估报告和案例分析结果，默认不提交 Git。

常见目录：

- `training/`
- `training_pairwise/`
- `eval_reports/`
- `eval_reports_four_way_q300/`
- `data_checks/`

生成四方离线评估报告：

```powershell
python backend/scripts/run_evaluation.py --query-templates augmented --max-queries 300 --include-trained-reranker --include-pairwise-reranker --output-dir outputs/eval_reports_four_way_q300
```
