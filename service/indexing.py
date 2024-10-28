
from opensearchpy import OpenSearch


# 인덱스 생성 및 데이터 인덱싱
host = 'localhost'
port = 9200
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=('admin', 'admin'),
    use_ssl=False
)

model_id = ""

pipeline = "bert-pipeline"
if not client.ingest.get_pipeline(pipeline):
    pipeline_body = {
        "description": "Extracts embeddings from text using a pre-trained model",
        "processors": [
            {
                "text_embedding": {
                    "model_id": f"{model_id}",
                    "field_map": {
                        "review": "review_vector"
                    }
                }
            }
        ]
    }
    client.ingest.put_pipeline(pipeline, pipeline_body)
    print(f"파이프라인 '{pipeline}'이 생성되었습니다.")

index_name = "restaurant_recommendation"
index_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "default_pipeline": pipeline
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
    # 데이터 인덱싱
    response = client.index(index=index_name, body=restaurant_data)

    print(f"리뷰: {restaurant_data['review']}  인덱싱 임베딩 벡터: {restaurant_data['review_vector']}")
    if response['result'] == 'created':
        print(f"'{restaurant_data['shopName']}' 데이터가 인덱싱되었습니다.")
    else:
        print(f"'{restaurant_data['shopName']}' 데이터 인덱싱 실패:", response)
