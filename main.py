import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from opensearchpy import OpenSearch

# 1. 한국어 BERT 모델 로드
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


# 2. OpenSearch 클라이언트 설정 및 인덱스 생성
host = 'localhost'
port = 9200
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=('admin', 'admin'),
    use_ssl=False
)

index_name = "restaurant_recommendation"
index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {
            "knn": True
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
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",  # 또는 "l2"
                    "engine": "lucene",  # 또는 "faiss", "nmslib"
                    "parameters": {}
                }
            }
        }
    }
}

if not client.indices.exists(index_name):
    client.indices.create(index=index_name, body=index_body)
    print(f"인덱스 '{index_name}'가 생성되었습니다.")
else:
    print(f"인덱스 '{index_name}'가 이미 존재합니다.")

# 3. 데이터 벡터화 및 인덱싱
restaurant_data_list = [
    {
        "shopName": "백소정",
        "location": "대륭포스트 타워6차 1층",
        "menu": [
            {"name": "등심 돈까스", "price": 12900},
            {"name": "안심 돈까스", "price": 13900},
            {"name": "마제소바", "price": 10000}
        ],
        "review": "점심 시간에 먹기 좋은 일본식 돈까스 집 특히 마제소바도 맛있음"
    },
    {
        "shopName": "맥도날드",
        "location": "대륭포스트 타워6차 1층",
        "menu": [
            {"name": "빅맥", "price": 5000},
            {"name": "1955버거", "price": 7000},
            {"name": "맥너겟", "price": 4000}
        ],
        "review": "최고의 햄버거 맛집은 역시 맥도날드"
    },
    {
        "shopName": "스타벅스",
        "location": "대륭포스트 타워6차 1층",
        "menu": [
            {"name": "아메리카노", "price": 4100},
            {"name": "카페라떼", "price": 4600},
            {"name": "카라멜 마키아또", "price": 5600}
        ],
        "review": "커피가 맛있는 스타벅스"
    },
    {
        "shopName": "맘스터치",
        "location": "대륭포스트 타워6차 1층",
        "menu": [
            {"name": "싸이버거", "price": 5000},
            {"name": "싸이버거세트", "price": 7000},
            {"name": "후렌치 후라이", "price": 3000}
        ],
        "review": "치킨버거의 최고 존엄 역시 맛있음"
    },
]

# 데이터 인덱싱
for restaurant_data in restaurant_data_list:
    # 리뷰 텍스트를 임베딩 벡터로 변환
    review_vector = get_sentence_embedding(restaurant_data["review"])
    # 벡터를 리스트로 변환하여 JSON으로 전달 가능하게 함
    restaurant_data["review_vector"] = review_vector.tolist()
    # 데이터 인덱싱
    response = client.index(index=index_name, body=restaurant_data)

    print(f"리뷰: {restaurant_data['review']}  인덱싱 임베딩 벡터: {restaurant_data['review_vector']}")
    if response['result'] == 'created':
        print(f"'{restaurant_data['shopName']}' 데이터가 인덱싱되었습니다.")
    else:
        print(f"'{restaurant_data['shopName']}' 데이터 인덱싱 실패:", response)


# 4. 검색 기능 구현
def search_restaurants(query, top_k=3):
    # 검색 질의를 임베딩 벡터로 변환
    query_vector = get_sentence_embedding(query)
    # 벡터를 리스트로 변환
    query_vector = query_vector.tolist()

    print(f"검색 질의: {query}, 벡터: {query_vector}")

    # 검색 쿼리 작성
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "review_vector": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    # 검색 수행
    response = client.search(index=index_name, body=search_query)

    # 검색 결과 반환
    return response['hits']['hits']


# 예시 검색
query = "치킨버거가 맛있는 음식점 추천해줘"
results = search_restaurants(query)

for result in results:
    source = result['_source']
    score = result['_score'] - 1.0  # 실제 코사인 유사도 계산을 위해 1.0을 뺌
    print(f"식당 이름: {source['shopName']}, 리뷰: {source['review']}, 유사도: {score:.4f}")
