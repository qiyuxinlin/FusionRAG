#!/usr/bin/env python3
import pandas as pd

df_old = pd.read_csv('/mnt/data/reflect/Qwen2.5-7B-Instruct/results/musique/DraftModel_global_topk_10_rate_0.2_draft_Qwen2.5-3B-Instruct.csv')
df_new = pd.read_csv('/mnt/data/reflect/Qwen2.5-7B-Instruct/musique/results/DynamicDraftModel_global_topk_10_rate_0.2_draft_Qwen2.5-3B-Instruct.csv')

df_old['key'] = df_old['Main Question'] + '|||' + df_old['Sub Question']
df_new['key'] = df_new['Main Question'] + '|||' + df_new['Sub Question']

merged = df_old[['key', 'Sub Question', 'Predicted', 'Correct']].merge(
    df_new[['key', 'Predicted', 'Correct']], on='key', suffixes=('_old', '_new')
)

merged['same_prediction'] = merged['Predicted_old'].str.strip().str.lower() == merged['Predicted_new'].str.strip().str.lower()
merged['Correct_old'] = merged['Correct_old'].astype(str).str.lower().isin(['true', 'yes', '1'])
merged['Correct_new'] = merged['Correct_new'].astype(str).str.lower().isin(['true', 'yes', '1'])

same_pred_diff_judge = merged[merged['same_prediction'] & (merged['Correct_old'] != merged['Correct_new'])]
print(f'预测答案相同但 judge 结果不同: {len(same_pred_diff_judge)} 个')

diff_pred = merged[~merged['same_prediction']]
print(f'预测答案不同: {len(diff_pred)} 个')

diff_pred_old_correct = diff_pred[diff_pred['Correct_old'] & ~diff_pred['Correct_new']]
diff_pred_new_correct = diff_pred[~diff_pred['Correct_old'] & diff_pred['Correct_new']]
print(f'  其中 old 对 new 错: {len(diff_pred_old_correct)} 个')
print(f'  其中 old 错 new 对: {len(diff_pred_new_correct)} 个')

print()
print('=== 答案不同且 new 对了 ===')
for i, (_, row) in enumerate(diff_pred_new_correct.head(5).iterrows()):
    print(f'{i+1}. Q: {row["Sub Question"][:60]}')
    print(f'   old: {row["Predicted_old"][:50]}')
    print(f'   new: {row["Predicted_new"][:50]}')
    print()
