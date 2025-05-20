import json
import csv

# JSON 파일 경로
input_path = '/home/user16/HT/Video/LLoVi_implementation/output/egoschema/hcqa_global_full.json'
output_path = '/home/user16/HT/Video/LLoVi_implementation/output/egoschema/gpt3.5_global_full_submission.csv'

# JSON 파일 불러오기
with open(input_path) as f:
    a = json.load(f)['data']

# CSV 파일로 변환
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['q_uid', 'prediction'])  # 헤더

    for key, value in a.items():
        pred = int(value['pred'])  # 정수로 변환
        writer.writerow([key, pred])