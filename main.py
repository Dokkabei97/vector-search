from transformers import AutoTokenizer, AutoModel
import torch

model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 샘플 입력 생성
dummy_input = tokenizer("안녕하세요", return_tensors="pt")

# 모델을 ONNX로 변환하여 저장
torch.onnx.export(
    model,
    args=(
        dummy_input['input_ids'],
        dummy_input['attention_mask'],
        dummy_input.get('token_type_ids')
    ),
    f="korean_bert.onnx",
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size', 1: 'sequence'}
    },
    opset_version=14
)

from opensearchpy import OpenSearch

# OpenSearch 클라이언트 설정
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    # http_auth=('username', 'password'),  # 인증 정보 설정
    # use_ssl=True,
    # verify_certs=False,
    # ssl_show_warn=False
)

from opensearch_py_ml.ml_commons import MLCommonClient

ml_client = MLCommonClient(client)

# 모델 그룹 생성 요청
model_group_request = {
    "name": "korean_bert_group",
    "description": "Korean BERT Model Group",
    "model_access_mode": "public"  # 또는 "private", "restricted"
}

# 모델 그룹 생성 API 호출
response = client.transport.perform_request(
    method='POST',
    url='/_plugins/_ml/model_groups/_register',
    body=model_group_request
)

# 모델 그룹 ID 추출
model_group_id = response['model_group_id']
print(f"Model Group ID: {model_group_id}")

# 모델 등록 요청
model_register_request = {
    "name": "korean_bert",
    "version": "1.0",
    "model_format": "ONNX",
    "model_group_id": model_group_id,
    "description": "Korean BERT ONNX model",
    "model_config": {
        "model_type": "bert",
        "embedding_dimension": 768,
        "framework_type": "sentence_transformers"
    }
}

# 모델 등록 API 호출
register_response = client.transport.perform_request(
    method='POST',
    url='/_plugins/_ml/models/_register',
    body=model_register_request
)

# 응답 확인
print(f"Register Response: {register_response}")

# model_id와 task_id 추출
model_id = register_response.get('model_id')
task_id = register_response.get('task_id')
print(f"Model ID: {model_id}")
print(f"Task ID: {task_id}")

import os
import math
import base64

def upload_model_chunks(model_id, model_path):
    # 모델 파일 크기 확인
    file_size = os.path.getsize(model_path)
    chunk_size = 10 * 1024 * 1024  # 10MB
    total_chunks = math.ceil(file_size / chunk_size)

    with open(model_path, 'rb') as f:
        for chunk_number in range(total_chunks):
            chunk_data = f.read(chunk_size)
            # 모델 청크 업로드 요청
            chunk_request = {
                "model_id": model_id,
                "chunk_number": chunk_number,
                "total_chunks": total_chunks,
                "model_content": base64.b64encode(chunk_data).decode('utf-8')  # 바이너리를 Base64로 인코딩
            }
            response = client.transport.perform_request(
                method='POST',
                url=f"/_plugins/_ml/models/{model_id}/chunk",
                body=chunk_request
            )
            print(f"Uploaded chunk {chunk_number + 1}/{total_chunks}")

# 모델 청크 업로드 실행
upload_model_chunks(model_id, "korean_bert.onnx")

import time

def wait_for_model_registration(task_id):
    status_url = f"/_plugins/_ml/tasks/{task_id}"
    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        try:
            status_response = client.transport.perform_request(
                method='GET',
                url=status_url
            )
            print("Status Response:", status_response)

            state = status_response.get('state', '')
            if state == 'COMPLETED':
                print("Model registration completed.")
                return
            elif state == 'FAILED':
                error_msg = status_response.get('error', {})
                print("Detailed error:", error_msg)
                raise Exception(f"Model registration failed: {error_msg}")
            else:
                print(f"Task state: {state}, waiting... (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(10)  # 10초 대기

        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(5)

        attempt += 1

    raise Exception("Maximum attempts reached while waiting for model registration")

# 모델 등록 작업 완료 대기
wait_for_model_registration(task_id)

# 모델 배포 요청
deploy_request = {
    "model_id": model_id
}

deploy_response = client.transport.perform_request(
    method='POST',
    url=f"/_plugins/_ml/models/{model_id}/deploy",
    body=deploy_request
)

print(f"Deploy Response: {deploy_response}")

index_name = "restaurant_index"

index_mapping = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True  # KNN을 활성화해야 합니다.
        }
    },
    "mappings": {
        "properties": {
            "shopName": {"type": "text"},
            "location": {"type": "text"},
            "menu": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "price": {"type": "integer"}
                }
            },
            "review": {"type": "text"},
            "review_vector": {
                "type": "knn_vector",
                "dimension": 768  # BERT 임베딩 차원과 일치
            }
        }
    }
}

# 인덱스 생성
client.indices.create(index=index_name, body=index_mapping)

import numpy as np

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 문장 임베딩 생성 (평균 풀링)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# 예제 데이터
doc = {
    "shopName": "백소정",
    "location": "대륭포스타 타워6차 1층",
    "menu": [
        {"name": "등심 돈까스", "price": 12900},
        {"name": "안심 돈까스", "price": 13900},
        {"name": "마제소바", "price": 10000}
    ],
    "review": "점심 시간에 먹기 좋은 일본식 돈까스 집 특히 마제소바도 맛있음"
}

# 리뷰 임베딩 생성
doc["review_vector"] = get_embedding(doc["review"]).tolist()

# 문서 인덱싱
client.index(index=index_name, body=doc)

def search(query, index_name, k=3):
    # 쿼리 임베딩 생성
    query_vector = get_embedding(query).tolist()

    # 검색 쿼리 작성
    search_query = {
        "size": k,
        "query": {
            "knn": {
                "review_vector": {
                    "vector": query_vector,
                    "k": k
                }
            }
        }
    }

    # 검색 수행
    response = client.search(index=index_name, body=search_query)
    return response

# 예제 쿼리
user_query = "맛있는 돈까스 집 추천해줘"

# 검색 실행
results = search(user_query, index_name)

# 결과 출력
for hit in results['hits']['hits']:
    print(f"Score: {hit['_score']}, Shop Name: {hit['_source']['shopName']}, Review: {hit['_source']['review']}")