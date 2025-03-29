import hashlib
import time

from config.opensearch_config import setup_opensearch_client


# 모델 ZIP 파일의 SHA256 체크섬 계산
def calculate_sha256(file_path):
    print('============== calculate_sha256 ============== ')
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    print(f'SHA256: {sha256_hash.hexdigest()}')
    return sha256_hash.hexdigest()


# 클러스터 설정 변경
def cluster_settings(client):
    print('============== cluster_settings ============== ')
    settings = {
        "persistent": {
            "plugins.ml_commons.allow_registering_model_via_url": "true",
            "plugins.ml_commons.only_run_on_ml_node": "false",
            "plugins.ml_commons.model_access_control_enabled": "true",
            "plugins.ml_commons.native_memory_threshold": "99"
        }
    }

    response = client.cluster.put_settings(body=settings)
    print(f"Cluster settings: {response}")


# 모델 그룹 생성
def create_model_group(client, name, description):
    print('============== create_model_group ============== ')
    model_group_request = {
        "name": f"{name}-15",
        "description": description,
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
def register_model(client, name, version, description, model_format, model_group_id, model_content_hash_value,
                   model_config, model_url):
    print('============== register_model ============== ')
    model_register_request = {
        "name": name,
        "version": version,  # 문자열 형태
        "description": description,
        "model_format": model_format,  # "ONNX" 또는 "TORCH_SCRIPT"
        "model_group_id": model_group_id,
        "model_content_hash_value": model_content_hash_value,
        "model_config": model_config,
        "url": model_url  # 호스팅된 ZIP 파일의 URL
    }
    print(f"Model Register Request: {model_register_request}")

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
    print('============== get_model_id_from_task ============== ')
    status_url = f"/_plugins/_ml/tasks/{task_id}"
    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        status_response = client.transport.perform_request(
            method='GET',
            url=status_url
        )
        print("Status Response:", status_response)
        try:
            state = status_response.get('state', '')
            if state == 'COMPLETED':
                model_id = status_response.get('model_id')
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
                print(f"Task state: {state}, waiting... 20 second (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(20)  # 15초 대기

        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(5)

        attempt += 1

    raise Exception("Maximum attempts reached while waiting for model registration.")


# 7. 모델 배포
def deploy_model(client, model_id):
    print('============== deploy_model ============== ')
    deploy_response = client.transport.perform_request(
        method='POST',
        url=f"/_plugins/_ml/models/{model_id}/_deploy",
    )
    print(f"Deploy Response: {deploy_response}")

    task_id = deploy_response.get('task_id')

    max_attempts = 10
    attempt = 0
    while attempt < max_attempts:
        status_response = client.transport.perform_request(
            method='GET',
            url=f"/_plugins/_ml/tasks/{task_id}"
        )
        print("Status Response:", status_response)
        try:
            state = status_response.get('state', '')
            if state == 'COMPLETED':
                print("Model deployment completed.")
                return
            elif state == 'FAILED':
                error_msg = status_response.get('error', 'Unknown error')
                print("Detailed error:", error_msg)
                raise Exception(f"Model deployment failed: {error_msg}")
            else:
                print(f"Task state: {state}, waiting... 20 second (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(20)  # 15초 대기

        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(5)

        attempt += 1

# 11. 전체 프로세스 실행
def main():
    # 1. 모델 변환 및 ZIP 파일 생성
    model_zip_path = "../model/kbert.zip"

    # 2. 모델 ZIP 파일의 SHA256 체크섬 계산
    model_content_hash_value = calculate_sha256(model_zip_path)
    print(f"Model ZIP SHA256: {model_content_hash_value}")

    # 3. OpenSearch 클라이언트 설정
    client = setup_opensearch_client()
    cluster_settings(client)

    # 4. 모델 그룹 생성
    model_group_id = create_model_group(
        client,
        name="korean_bert_group",
        description="Korean BERT Model Group",
    )

    # 5. 모델 등록
    model_zip_url = "http://localhost:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL3NoYXJlZC1vYmplY3Qva2JlcnQuemlwP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9ODRXNjkyREMySjAwS1NNNlpUM0QlMkYyMDI1MDMyMyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTAzMjNUMDYxMjQzWiZYLUFtei1FeHBpcmVzPTQzMTk5JlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lJNE5GYzJPVEpFUXpKS01EQkxVMDAyV2xRelJDSXNJbVY0Y0NJNk1UYzBNamMxTURZd09Td2ljR0Z5Wlc1MElqb2liV2x1YVc5a2IyTnJaWElpZlEuZ0ZiTjRLUDBoLWJEa0J6T3hpUFlCNzRxWlhwSkRTWnQyeXc4bjg2dHpNc2lkRVdLZ2s4eWVWeUJkaHpEeVktSVc3ZkxtekNxN3dPbHZBUDctY0Zha2cmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnZlcnNpb25JZD1udWxsJlgtQW16LVNpZ25hdHVyZT1jMmUwYmJlOWJhNjRlZThiNzQ1OGQ2YjkwNjJmNmUxZDM1Y2U3OWZkYjUzNjBiZDE0YTFjMzc0NGYwYTI4MGZj"

    task_id = register_model(
        client,
        name="korean_bert",
        version="1.0.0",
        description="Korean BERT ONNX model",
        model_format="ONNX",
        model_group_id=model_group_id,
        model_content_hash_value=f"{model_content_hash_value}",
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
        model_url=model_zip_url.replace("localhost", "minio")
    )

    # 6. 모델 등록 작업 완료 대기 및 model_id 추출
    try:
        model_id = get_model_id_from_task(client, task_id)
    except Exception as e:
        print(f"Error during model registration: {e}")
        return

    # 7. 모델 배포
    deploy_model(client, model_id)


if __name__ == "__main__":
    main()
