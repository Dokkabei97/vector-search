import hashlib
import os
import time
import zipfile

import torch
from opensearchpy import OpenSearch
from transformers import AutoTokenizer, AutoModel


# 1. BERT 모델을 ONNX로 변환하고 tokenizer.json 포함하여 ZIP 파일 생성
def convert_model_to_onnx(model_name, output_path, opset_version=14):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 샘플 입력 생성
    dummy_input = tokenizer("안녕하세요", return_tensors="pt")

    # 모델을 ONNX로 변환하여 저장
    torch.onnx.export(
        model,
        args=(
            dummy_input['input_ids'],
            dummy_input['attention_mask']
            # 'token_type_ids'가 필요한 경우 추가
        ),
        f="korean_bert.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=opset_version
    )
    print(f"Model converted to ONNX and saved at 'korean_bert.onnx'")

    # tokenizer.json 저장
    tokenizer.save_pretrained(".")
    print("Tokenizer files saved.")

    # 모델 파일과 tokenizer.json을 ZIP으로 압축
    with zipfile.ZipFile(output_path, 'w') as zipf:
        zipf.write("korean_bert.onnx")
        # tokenizer.json 및 관련 파일들 추가
        for file_name in ["tokenizer.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]:
            if os.path.exists(file_name):
                zipf.write(file_name)
                print(f"Added {file_name} to ZIP.")
            else:
                print(f"Warning: {file_name} does not exist and was not added to ZIP.")
    print(f"Model and tokenizer zipped into '{output_path}'")

# 2. 모델 ZIP 파일의 SHA256 체크섬 계산
def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 3. OpenSearch 클라이언트 설정
def setup_opensearch_client():
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        # 실제 환경에서는 아래 주석을 해제하고 인증 정보를 설정하세요.
        # http_auth=('username', 'password'),
        # use_ssl=True,
        # verify_certs=True,
        # ssl_show_warn=False
    )
    return client

# 4. 모델 그룹 생성
def create_model_group(client, name, description, access_mode):
    model_group_request = {
        "name": name,
        "description": description,
        "model_access_mode": access_mode  # "public", "private", "restricted"
    }

    response = client.transport.perform_request(
        method='POST',
        url='/_plugins/_ml/model_groups/_register',
        body=model_group_request
    )

    model_group_id = response.get('model_group_id')
    if not model_group_id:
        raise Exception("Failed to create model group.")
    print(f"Model Group ID: {model_group_id}")
    return model_group_id

# 5. 모델 등록
def register_model(client, name, version, model_format, model_group_id, description, function_name, model_content_size_in_bytes, model_content_hash_value, model_config, url):
    model_register_request = {
        "name": name,
        "version": version,  # 문자열 형태
        "model_format": model_format,  # "ONNX" 또는 "TORCH_SCRIPT"
        "model_group_id": model_group_id,
        "description": description,
        "function_name": function_name,  # "TEXT_EMBEDDING", "SPARSE_ENCODING", etc.
        "model_content_size_in_bytes": model_content_size_in_bytes,
        "model_content_hash_value": model_content_hash_value,
        "model_config": model_config,
        "url": url  # 호스팅된 ZIP 파일의 URL
    }

    register_response = client.transport.perform_request(
        method='POST',
        url='/_plugins/_ml/models/_register',
        body=model_register_request
    )

    task_id = register_response.get('task_id')
    if not task_id:
        raise Exception("Failed to register model.")
    print(f"Register Response: {register_response}")
    print(f"Task ID: {task_id}")
    return task_id

# 6. 모델 등록 작업 완료 대기 및 model_id 추출
def get_model_id_from_task(client, task_id):
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
                model_id = status_response['response'].get('model_id')
                if model_id:
                    print(f"Model ID: {model_id}")
                    return model_id
                else:
                    raise Exception("Model ID not found in the task response.")
            elif state == 'FAILED':
                error_msg = status_response.get('error', 'Unknown error')
                print("Detailed error:", error_msg)
                raise Exception(f"Model registration failed: {error_msg}")
            else:
                print(f"Task state: {state}, waiting... (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(10)  # 10초 대기

        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(5)

        attempt += 1

    raise Exception("Maximum attempts reached while waiting for model registration.")

# 7. 모델 배포
def deploy_model(client, model_id):
    deploy_request = {
        "model_id": model_id
    }

    deploy_response = client.transport.perform_request(
        method='POST',
        url=f"/_plugins/_ml/models/{model_id}/deploy",
        body=deploy_request
    )

    print(f"Deploy Response: {deploy_response}")

# 8. 인덱스 생성
def create_index(client, index_name, dimension):
    index_mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "knn": True  # KNN 활성화
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
                    "dimension": dimension  # 임베딩 차원
                }
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=index_mapping)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")

# 9. 데이터 인덱싱
def get_embedding(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 문장 임베딩 생성 (평균 풀링)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def index_document(client, tokenizer, model, index_name, doc):
    # 리뷰 임베딩 생성
    doc["review_vector"] = get_embedding(tokenizer, model, doc["review"]).tolist()

    # 문서 인덱싱
    try:
        client.index(index=index_name, body=doc)
        print("Document indexed successfully.")
    except Exception as e:
        print(f"Error indexing document: {e}")

# 10. 벡터 검색
def search(client, tokenizer, model, query, index_name, k=3):
    # 쿼리 임베딩 생성
    query_vector = get_embedding(tokenizer, model, query).tolist()

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
    try:
        response = client.search(index=index_name, body=search_query)
        return response
    except Exception as e:
        print(f"Error performing search: {e}")
        return None

# 11. 전체 프로세스 실행
def main():
    # 1. 모델 변환 및 ZIP 파일 생성
    model_name = "klue/bert-base"
    onnx_model_path = "korean_bert.onnx"
    model_zip_path = "korean_bert.zip"
    # convert_model_to_onnx(model_name, model_zip_path, opset_version=14)

    # 2. 모델 ZIP 파일의 SHA256 체크섬 계산
    # model_content_size_in_bytes = os.path.getsize(model_zip_path)
    # model_content_hash_value = calculate_sha256(model_zip_path)
    # print(f"Model ZIP Size: {model_content_size_in_bytes} bytes")
    # print(f"Model ZIP SHA256: {model_content_hash_value}")

    # 3. OpenSearch 클라이언트 설정
    client = setup_opensearch_client()

    # 4. 모델 그룹 생성
    model_group_id = create_model_group(
        client,
        name="korean_bert_group",
        description="Korean BERT Model Group",
        access_mode="public"  # "public", "private", "restricted"
    )

    # 5. 모델 등록
    # 호스팅된 모델 ZIP 파일 URL 설정
    model_zip_url = "https://github.com/Dokkabei97/vector-search/releases/download/vector-model/korean_bert.zip"  # 실제 호스팅된 URL로 변경

    task_id = register_model(
        client,
        name="korean_bert",
        version="1.0.0",
        model_format="ONNX",
        model_group_id=model_group_id,
        description="Korean BERT ONNX model",
        function_name="TEXT_EMBEDDING",
        model_content_size_in_bytes="443756946",
        model_content_hash_value="1098f7a1608d01a6db6f4fe9b7b3314390a3454c7abf3ad51b8982a868375482",
        model_config={
            "model_type": "bert",
            "embedding_dimension": 768,
            "framework_type": "sentence_transformers",
            "all_config": """{
                "_name_or_path": "klue/bert-base",
                "architectures": ["BertModel"],
                "attention_probs_dropout_prob": 0.1,
                "gradient_checkpointing": false,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "transformers_version": "4.11.3",
                "type_vocab_size": 2,
                "use_cache": true,
                "vocab_size": 30522
            }"""
        },
        url=model_zip_url
    )

    # 6. 모델 등록 작업 완료 대기 및 model_id 추출
    try:
        model_id = get_model_id_from_task(client, task_id)
    except Exception as e:
        print(f"Error during model registration: {e}")
        return

    # 7. 모델 배포
    deploy_model(client, model_id)

    # 8. 인덱스 생성
    index_name = "restaurant_index"
    create_index(client, index_name, dimension=768)

    # 9. 데이터 인덱싱
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

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

    index_document(client, tokenizer, model, index_name, doc)

    # 10. 벡터 검색
    user_query = "맛있는 돈까스 집 추천해줘"
    results = search(client, tokenizer, model, user_query, index_name)

    # 결과 출력
    if results:
        for hit in results['hits']['hits']:
            print(f"Score: {hit['_score']}, Shop Name: {hit['_source']['shopName']}, Review: {hit['_source']['review']}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()