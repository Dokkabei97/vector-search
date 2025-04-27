# src/model_deploy.py

import hashlib
import time
import logging
import json # all_config를 JSON으로 로드하기 위해 추가

# opensearch_config 모듈이 같은 디렉토리에 있다고 가정
from src.opensearch_config import setup_opensearch_client

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 변경 부분 ---
# 변환된 모델 ZIP 파일 경로 (model_converter.py에서 생성된 경로와 일치)
MODEL_ZIP_PATH = "../model/klue_bert_base.zip"
# OpenSearch에 등록할 모델 이름
MODEL_NAME = "klue-bert-base"
# OpenSearch에 등록할 모델 버전
MODEL_VERSION = "1.0.0"
# OpenSearch 모델 그룹 이름 (필요시 변경)
MODEL_GROUP_NAME = "korean-language-models"
# OpenSearch 모델 그룹 설명 (필요시 변경)
MODEL_GROUP_DESCRIPTION = "Group for Korean NLP models"
# 모델 포맷 (ONNX 또는 TORCH_SCRIPT)
MODEL_FORMAT = "ONNX"
# klue/bert-base의 임베딩 차원
EMBEDDING_DIMENSION = 768
# 사용할 프레임워크 타입 (Sentence Transformers 권장)
FRAMEWORK_TYPE = "sentence_transformers" # OpenSearch가 풀링을 처리하도록 함
# 모델 타입
MODEL_TYPE = "bert"

# !!! 중요 !!!
# MinIO 또는 사용하는 스토리지에 klue_bert_base.zip 파일을 업로드한 후,
# 해당 파일에 접근 가능한 **실제 URL**로 아래 값을 반드시 수정해야 합니다.
# 예시 URL이며, 실제 환경에 맞게 변경 필요
MODEL_ZIP_URL = "http://minio:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL3NoYXJlZC1vYmplY3Qva2x1ZV9iZXJ0X2Jhc2UuemlwP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9RzdFSVdPSjA2RzRKM09WOVkwWVolMkYyMDI1MDQyNyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA0MjdUMDQzMjAxWiZYLUFtei1FeHBpcmVzPTQzMTk5JlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lKSE4wVkpWMDlLTURaSE5Fb3pUMVk1V1RCWldpSXNJbVY0Y0NJNk1UYzBOVGMzTVRBek1Dd2ljR0Z5Wlc1MElqb2liV2x1YVc5a2IyTnJaWElpZlEuVDNYOEtEdElmaGJwQ0NZSzZFU1lsWFo3V3RST083X1FtMHNXd3RCWG5DTTRQU0VDaFhqOFk1ODBlaHBDZVpfVFU4UHJJakNPTkJOTUJyMldDcTlPQUEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnZlcnNpb25JZD1udWxsJlgtQW16LVNpZ25hdHVyZT0wZjY1MTg0MjFkOGJhNTY3OGM1MDYzMDQ2ZDMxNTRmODRkZWM5MGViNjI1MThiMGIwNzk0NzBhZDM5M2M4NGI0" # <--- 실제 URL로 변경하세요!
# --- 설정 변경 끝 ---


def calculate_sha256(file_path):
    """파일의 SHA256 체크섬을 계산합니다."""
    logging.info(f"Calculating SHA256 checksum for: {file_path}")
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # 파일을 작은 블록으로 나누어 읽어 메모리 사용량 관리
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        hex_digest = sha256_hash.hexdigest()
        logging.info(f"SHA256 Checksum: {hex_digest}")
        return hex_digest
    except FileNotFoundError:
        logging.error(f"Error: Model ZIP file not found at {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error calculating SHA256: {e}")
        raise

def configure_cluster_settings(client):
    """OpenSearch 클러스터 설정을 구성합니다 (모델 등록 허용 등)."""
    logging.info("Configuring OpenSearch cluster settings for ML Commons.")
    settings = {
        "persistent": {
            # URL을 통한 모델 등록 허용
            "plugins.ml_commons.allow_registering_model_via_url": "true",
            # ML 노드에서만 모델 실행 여부 (개발 환경에서는 false로 설정 가능)
            "plugins.ml_commons.only_run_on_ml_node": "false",
            # 모델 접근 제어 활성화
            "plugins.ml_commons.model_access_control_enabled": "true",
            # 네이티브 메모리 임계값 (필요시 조정)
            "plugins.ml_commons.native_memory_threshold": "99"
        }
    }
    try:
        response = client.cluster.put_settings(body=settings)
        logging.info(f"Cluster settings update response: {response}")
    except Exception as e:
        logging.error(f"Failed to update cluster settings: {e}")
        raise

def create_or_get_model_group(client, name, description):
    """지정된 이름의 모델 그룹을 생성하거나 기존 그룹 ID를 반환합니다."""
    logging.info(f"Checking or creating model group: {name}")
    model_group_request = {
        "name": name,
        "description": description,
    }
    try:
        # 모델 그룹 등록 시도
        response = client.transport.perform_request(
            method='POST',
            url='/_plugins/_ml/model_groups/_register',
            body=model_group_request
        )
        model_group_id = response.get('model_group_id')
        if not model_group_id:
            raise Exception("Failed to get model_group_id from registration response.")
        logging.info(f"Model Group registered with ID: {model_group_id}")
        return model_group_id
    except Exception as e:
        # 이미 존재하는 경우 오류 메시지에서 그룹 ID 추출 시도 (주의: OpenSearch 버전에 따라 오류 메시지 형식이 다를 수 있음)
        if "model group already exists" in str(e).lower():
            logging.warning(f"Model group '{name}' already exists. Attempting to find its ID.")
            try:
                # 모든 모델 그룹 검색
                search_response = client.transport.perform_request(
                    method='GET',
                    url=f'/_plugins/_ml/model_groups/_search?q=name:{name}'
                )
                hits = search_response.get('hits', {}).get('hits', [])
                if hits:
                    model_group_id = hits[0].get('_id')
                    if model_group_id:
                        logging.info(f"Found existing Model Group ID: {model_group_id}")
                        return model_group_id
                raise Exception(f"Model group '{name}' exists but failed to retrieve its ID.")
            except Exception as search_e:
                logging.error(f"Failed to retrieve existing model group ID: {search_e}")
                raise search_e
        else:
            logging.error(f"Failed to register model group: {e}")
            raise e


def register_model(client, name, version, description, model_format, model_group_id,
                   model_content_hash_value, model_config, model_url):
    """모델을 OpenSearch ML Commons에 등록합니다."""
    logging.info(f"Registering model: {name} (Version: {version})")
    model_register_request = {
        "name": name,
        "version": version,
        "description": description,
        "model_format": model_format,
        "model_group_id": model_group_id,
        "model_content_hash_value": model_content_hash_value,
        "model_config": model_config,
        "url": model_url # 호스팅된 ZIP 파일의 URL
    }
    logging.debug(f"Model Registration Request Body: {json.dumps(model_register_request, indent=2)}")

    try:
        register_response = client.transport.perform_request(
            method='POST',
            url='/_plugins/_ml/models/_register',
            body=model_register_request
        )
        task_id = register_response.get('task_id')
        if not task_id:
            raise Exception("Failed to get task_id from registration response.")
        logging.info(f"Model registration task submitted. Task ID: {task_id}")
        return task_id
    except Exception as e:
        logging.error(f"Failed to submit model registration task: {e}")
        logging.error(f"Request body was: {json.dumps(model_register_request, indent=2)}") # 오류 시 요청 본문 로깅
        raise

def wait_for_task_completion(client, task_id, task_description="Task"):
    """주어진 Task ID가 완료될 때까지 대기하고 결과를 반환합니다."""
    logging.info(f"Waiting for {task_description} (Task ID: {task_id}) to complete...")
    status_url = f"/_plugins/_ml/tasks/{task_id}"
    max_attempts = 20 # 최대 시도 횟수 증가
    wait_interval = 15 # 대기 시간 (초)

    for attempt in range(max_attempts):
        try:
            status_response = client.transport.perform_request(
                method='GET',
                url=status_url
            )
            state = status_response.get('state', '').upper()
            logging.info(f"Attempt {attempt + 1}/{max_attempts}: Task state: {state}")

            if state == 'COMPLETED':
                logging.info(f"{task_description} completed successfully.")
                return status_response # 완료 시 전체 응답 반환
            elif state == 'FAILED':
                error_msg = status_response.get('error', 'Unknown error')
                logging.error(f"{task_description} failed: {error_msg}")
                logging.error(f"Full failure response: {status_response}")
                raise Exception(f"{task_description} failed: {error_msg}")
            elif state in ['CREATED', 'RUNNING', 'REGISTERING', 'DEPLOYING']:
                # 진행 중인 상태, 대기
                pass
            else:
                logging.warning(f"Unknown task state encountered: {state}")
                # 알 수 없는 상태도 일단 대기

            time.sleep(wait_interval)

        except Exception as e:
            # API 호출 자체에서 오류 발생 시
            logging.error(f"Error checking task status for {task_id}: {e}")
            # 몇 번 더 시도해 볼 수 있도록 잠시 대기 후 계속 진행
            if attempt < max_attempts - 1:
                time.sleep(wait_interval)
            else:
                raise # 마지막 시도에서도 오류 발생 시 예외 발생

    # 최대 시도 횟수 도달
    raise TimeoutError(f"Maximum attempts reached while waiting for {task_description} (Task ID: {task_id}).")


def deploy_model(client, model_id):
    """등록된 모델을 배포합니다."""
    logging.info(f"Deploying model with ID: {model_id}")
    try:
        deploy_response = client.transport.perform_request(
            method='POST',
            url=f"/_plugins/_ml/models/{model_id}/_deploy",
            # body={} # 필요시 배포 관련 파라미터 추가 가능
        )
        task_id = deploy_response.get('task_id')
        if not task_id:
            raise Exception("Failed to get task_id from deployment response.")
        logging.info(f"Model deployment task submitted. Task ID: {task_id}")
        return task_id
    except Exception as e:
        logging.error(f"Failed to submit model deployment task for model ID {model_id}: {e}")
        raise

def main():
    """메인 실행 함수"""
    try:
        # 1. 모델 ZIP 파일의 SHA256 체크섬 계산
        model_content_hash_value = calculate_sha256(MODEL_ZIP_PATH)

        # 2. OpenSearch 클라이언트 설정
        client = setup_opensearch_client()

        # 3. 클러스터 설정 변경 (필요한 경우 한 번만 실행)
        configure_cluster_settings(client)

        # 4. 모델 그룹 생성 또는 ID 가져오기
        model_group_id = create_or_get_model_group(
            client,
            name=MODEL_GROUP_NAME,
            description=MODEL_GROUP_DESCRIPTION,
        )

        # 5. 모델 등록 정보 구성
        # framework_type='sentence_transformers' 사용 시 all_config는 불필요하거나 단순화 가능
        model_config = {
            "model_type": MODEL_TYPE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "framework_type": FRAMEWORK_TYPE,
            # "all_config": "{...}" # 필요 시 klue/bert-base의 config.json 내용을 문자열로 넣거나 제거
        }

        # 6. 모델 등록 작업 제출
        register_task_id = register_model(
            client,
            name=MODEL_NAME,
            version=MODEL_VERSION,
            description=f"{MODEL_NAME} ONNX model ({FRAMEWORK_TYPE})",
            model_format=MODEL_FORMAT,
            model_group_id=model_group_id,
            model_content_hash_value=model_content_hash_value,
            model_config=model_config,
            model_url=MODEL_ZIP_URL # !!! 이 URL이 올바른지 확인하세요 !!!
        )

        # 7. 모델 등록 작업 완료 대기 및 model_id 추출
        registration_status = wait_for_task_completion(client, register_task_id, "Model Registration")
        model_id = registration_status.get('model_id')
        if not model_id:
            raise Exception("Model ID not found in the completed registration task response.")
        logging.info(f"Model registered successfully. Model ID: {model_id}")

        # 8. 모델 배포 작업 제출
        deploy_task_id = deploy_model(client, model_id)

        # 9. 모델 배포 작업 완료 대기
        wait_for_task_completion(client, deploy_task_id, "Model Deployment")
        logging.info(f"Model ID: {model_id} deployed successfully!")

    except FileNotFoundError:
        logging.error(f"Prerequisite error: Model file '{MODEL_ZIP_PATH}' not found.")
    except TimeoutError as e:
        logging.error(f"Operation timed out: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the deployment process: {e}")

if __name__ == "__main__":
    main()
