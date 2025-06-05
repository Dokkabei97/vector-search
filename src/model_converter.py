# src/model_converter.py

import os
import zipfile
import logging # 로깅 추가

# transformers 라이브러리 설치 필요: pip install transformers torch onnx
# sentencepiece 설치 필요: pip install sentencepiece (klue/bert-base는 WordPiece 기반이지만, transformers 설치 시 함께 관리됨)
from transformers import AutoTokenizer, AutoModel
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 변경 부분 ---
# 사용할 모델 이름 변경 (파인 튜닝된 로컬 모델 경로)
model_name = './fine_tuned_model'
# 출력될 ONNX 파일 이름
onnx_filename = "korean_finetuned.onnx"
# 출력될 ZIP 파일 이름
zip_filename = "korean_finetuned.zip"
# ZIP 파일 저장 경로 (상위 디렉토리의 model 폴더)
zip_output_path = os.path.join("..", "model", zip_filename)
# --- 설정 변경 끝 ---

def convert_model_to_onnx_and_zip():
    """
    지정된 Hugging Face 모델을 로드하고 ONNX로 변환한 후,
    모델과 토크나이저 파일을 ZIP으로 압축합니다.
    """
    try:
        logging.info(f"Loading model and tokenizer for: {model_name}")
        # AutoTokenizer와 AutoModel을 사용하여 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval() # 추론 모드로 설정

        logging.info("Creating dummy input for ONNX export.")
        # ONNX 변환을 위한 샘플 입력 생성
        dummy_input_text = "안녕하세요"
        dummy_input = tokenizer(dummy_input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # ONNX 변환 시 사용할 입력 이름 및 동적 축 설정
        input_names = list(dummy_input.keys())
        output_names = ['last_hidden_state'] # klue/bert-base의 기본 출력 중 하나
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: 'batch_size', 1: 'sequence_length'}
        dynamic_axes[output_names[0]] = {0: 'batch_size', 1: 'sequence_length'}

        # ONNX 변환 실행
        logging.info(f"Exporting model to ONNX format: {onnx_filename}")
        torch.onnx.export(
            model,                   # 실행될 모델
            tuple(dummy_input.values()), # 모델 입력값 (튜플 형태)
            f=onnx_filename,         # 저장될 ONNX 파일 경로
            input_names=input_names, # 모델 입력 이름 리스트
            output_names=output_names,# 모델 출력 이름 리스트
            dynamic_axes=dynamic_axes, # 동적 축 설정
            opset_version=14,        # ONNX Opset 버전
            export_params=True,      # 모델 파라미터 함께 저장
        )
        logging.info(f"Model successfully converted to ONNX: '{onnx_filename}'")

        # 토크나이저 파일 저장 (현재 디렉토리)
        logging.info("Saving tokenizer files to current directory.")
        tokenizer.save_pretrained(".")

        # ZIP 파일에 포함할 파일 목록 생성
        files_to_zip = [onnx_filename]
        # 토크나이저 관련 파일 목록 (일반적인 파일들)
        tokenizer_files = ["tokenizer.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json", "config.json"] # 모델 config.json도 포함
        for file_name in tokenizer_files:
            if os.path.exists(file_name):
                files_to_zip.append(file_name)
            else:
                logging.warning(f"Tokenizer file '{file_name}' not found, skipping.")

        # ZIP 파일 생성
        logging.info(f"Creating ZIP archive: '{zip_output_path}'")
        # ZIP 파일 저장 경로 생성 (폴더가 없다면)
        os.makedirs(os.path.dirname(zip_output_path), exist_ok=True)

        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files_to_zip:
                if os.path.exists(file):
                    # ZIP 파일 내부에 파일을 루트 레벨에 추가 (arcname 사용)
                    zipf.write(file, arcname=os.path.basename(file))
                    logging.info(f"Added '{file}' to ZIP archive.")
                else:
                    logging.warning(f"File '{file}' does not exist, cannot add to ZIP.")

        logging.info(f"Successfully created ZIP archive: '{zip_output_path}'")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        # 오류 발생 시 생성된 파일 일부를 삭제할 수 있음 (선택적)

    finally:
        # 임시 파일 정리 (ONNX 및 토크나이저 파일)
        logging.info("Cleaning up temporary files from current directory.")
        if os.path.exists(onnx_filename):
            try:
                os.remove(onnx_filename)
                logging.info(f"Removed temporary file: {onnx_filename}")
            except OSError as e:
                logging.error(f"Error removing file {onnx_filename}: {e}")
        for file_name in tokenizer_files:
            if os.path.exists(file_name):
                try:
                    os.remove(file_name)
                    logging.info(f"Removed temporary file: {file_name}")
                except OSError as e:
                    logging.error(f"Error removing file {file_name}: {e}")
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    convert_model_to_onnx_and_zip()
