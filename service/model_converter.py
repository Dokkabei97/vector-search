import os
import zipfile

from transformers import AutoTokenizer, AutoModel
import torch

# 모델 이름을 변경합니다.
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 샘플 입력 생성
dummy_input = tokenizer("안녕하세요", return_tensors="pt")

# token_type_ids가 있는지 확인 후, args 및 입력 이름 설정
if "token_type_ids" in dummy_input:
    args = (
        dummy_input['input_ids'],
        dummy_input['attention_mask'],
        dummy_input['token_type_ids']
    )
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size', 1: 'sequence'}
    }
else:
    args = (
        dummy_input['input_ids'],
        dummy_input['attention_mask']
    )
    input_names = ['input_ids', 'attention_mask']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size', 1: 'sequence'}
    }

# 모델을 ONNX로 변환하여 저장
torch.onnx.export(
    model,
    args=args,
    f="korean_bert.onnx",
    input_names=input_names,
    output_names=['output'],
    dynamic_axes=dynamic_axes,
    opset_version=14
)
print(f"Model converted to ONNX and saved at 'korean_bert.onnx'")

# 토크나이저 파일 저장
tokenizer.save_pretrained(".")
print("Tokenizer files saved.")

# 모델 파일과 관련 토크나이저 파일들을 ZIP으로 압축
with zipfile.ZipFile("../model/kbert.zip", 'w') as zipf:
    zipf.write("korean_bert.onnx")
    # tokenizer.json 및 관련 파일들 추가
    for file_name in ["tokenizer.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json", "unigram.json"]:
        if os.path.exists(file_name):
            zipf.write(file_name)
            print(f"Added {file_name} to ZIP.")
        else:
            print(f"Warning: {file_name} does not exist and was not added to ZIP.")
print(f"Model and tokenizer zipped into 'kbert.zip'")
