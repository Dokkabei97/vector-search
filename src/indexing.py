
from opensearchpy import OpenSearch


# 인덱스 생성 및 데이터 인덱싱
host = 'localhost'
port = 9200
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=('admin', 'admin'),
    use_ssl=False
)

model_id = "9T39wZUBycRyz7wDdYZJ"

pipeline = "bert-pipeline"
if not client.ingest.get_pipeline(pipeline):
    pipeline_body = {
        "description": "Extracts embeddings from text using a pre-trained model",
        "processors": [
            {
                "text_embedding": {
                    "model_id": f"{model_id}",
                    "field_map": {
                        "allDataSet": "totalIndex_vector"
                    }
                }
            }
        ]
    }
    client.ingest.put_pipeline(pipeline, pipeline_body)
    print(f"파이프라인 '{pipeline}'이 생성되었습니다.")

index_name = "catalog"
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
            "catalogName": {
                "type": "text",
                "copy_to": "totalIndex"
            },
            "brandName": {
                "type": "keyword",
                "copy_to": "totalIndex"
            },
            "brandCode": {"type": "keyword"},
            "makerCode": {"type": "keyword"},
            "makerName": {
                "type": "keyword",
                "copy_to": "totalIndex"
            },
            "catalogCreatedAt": {"type": "keyword"},
            "catalogModifiedAt": {"type": "keyword"},
            "catalogSeries": {
                "type": "nested",
                "properties": {
                    "catalogCode": {"type": "keyword"},
                    "price": {
                        "type": "keyword",
                        "copy_to": "totalIndex"
                    },
                    "modelCode": {"type": "keyword"},
                    "popularScore": {"type": "keyword"},
                    "catalogDescription": {
                        "type": "text",
                        "copy_to": "totalIndex"
                    },
                    "catalogOption": {
                        "type": "keyword",
                        "copy_to": "totalIndex"
                    },
                    "seriesCreatedAt": {"type": "keyword"},
                    "seriesModifiedAt": {"type": "keyword"}
                }
            },
            "totalIndex": {
                "type": "text",
                "store": True,
                "similarity": "boolean"
            },
            "allDataSet": {
                "type": "text"
            },
            "totalIndex_vector": {
                "type": "knn_vector",
                "dimension": 384,
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

shop_name = "abc"

# 3. 데이터 벡터화 및 인덱싱
catalog_data_list = [
    {
        "shopName": shop_name,
        "catalogName": "신라면 120g",
        "makerName": "농심",
        "makerCode": "10",
        "brandName": "신라면",
        "brandCode": "10",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "1",
                "price": "700",
                "modelCode": "1",
                "popularScore": "4.5",
                "catalogDescription": "봉지라면",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "2",
                "price": "7000",
                "modelCode": "2",
                "popularScore": "4.9",
                "catalogDescription": "봉지라면",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "3",
                "price": "21000",
                "modelCode": "3",
                "popularScore": "4.2",
                "catalogDescription": "봉지라면",
                "catalogOption": "30개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "4",
                "price": "35000",
                "modelCode": "4",
                "popularScore": "4.7",
                "catalogDescription": "봉지라면",
                "catalogOption": "50개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "신라면 65g",
        "makerName": "농심",
        "makerCode": "10",
        "brandName": "신라면",
        "brandCode": "10",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "5",
                "price": "800",
                "modelCode": "5",
                "popularScore": "4.9",
                "catalogDescription": "컵라면",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "6",
                "price": "7600",
                "modelCode": "6",
                "popularScore": "4.9",
                "catalogDescription": "컵라면",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "신라면 블랙 120g",
        "makerName": "농심",
        "makerCode": "10",
        "brandName": "신라면",
        "brandCode": "10",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "7",
                "price": "1000",
                "modelCode": "7",
                "popularScore": "4.9",
                "catalogDescription": "봉지라면",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "8",
                "price": "10000",
                "modelCode": "8",
                "popularScore": "4.9",
                "catalogDescription": "봉지라면",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "진라면 매운맛 120g",
        "makerName": "오뚜기",
        "makerCode": "11",
        "brandName": "진라면",
        "brandCode": "11",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "9",
                "price": "720",
                "modelCode": "9",
                "popularScore": "4.9",
                "catalogDescription": "봉지라면 / 매운맛",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "10",
                "price": "6900",
                "modelCode": "10",
                "popularScore": "4.6",
                "catalogDescription": "봉지라면 / 매운맛",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "진라면 순한맛 120g",
        "makerName": "오뚜기",
        "makerCode": "11",
        "brandName": "진라면",
        "brandCode": "11",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "11",
                "price": "720",
                "modelCode": "11",
                "popularScore": "4.9",
                "catalogDescription": "봉지라면 / 순한맛",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "12",
                "price": "6900",
                "modelCode": "12",
                "popularScore": "4.6",
                "catalogDescription": "봉지라면 / 순한맛",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "진라면 순한맛 70g",
        "makerName": "오뚜기",
        "makerCode": "11",
        "brandName": "진라면",
        "brandCode": "11",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "13",
                "price": "900",
                "modelCode": "13",
                "popularScore": "4.9",
                "catalogDescription": "컵라면 / 순한맛",
                "catalogOption": "1개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            },
            {
                "catalogCode": "14",
                "price": "8800",
                "modelCode": "14",
                "popularScore": "4.6",
                "catalogDescription": "컵라면 / 순한맛",
                "catalogOption": "10개",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "Z339NEEF11",
        "makerName": "LG",
        "makerCode": "13",
        "brandName": "오브제컬렉션",
        "brandCode": "13",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "20",
                "price": "1228770",
                "modelCode": "Z339NEEF11",
                "popularScore": "4.9",
                "catalogDescription": """
                    스탠드형/3도어/용량: 327L/2025년형/에너지: 1등급(24.09 기준)/
                    [디자인] 네이처(메탈)/베이지/
                    [용기] 용기: 4개/상칸: 미포함/중칸: 2개/하칸: 2개/용기용량: 56.8L/반투명김치통/
                    [냉각] 냉동겸용칸: 상칸(전체)/순환냉각/쿨링케어/냉기지킴가드/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채/과일, 육류/생선(생동), 쌀/잡곡/유산균김치/유산균인디케이터/
                    [편의] 탈취/칸칸전원/인버터/구입김치스캔보관/WiFi/스퀘어핸들/크기(가로x세로x깊이): 666x1787x737mm
                """,
                "catalogOption": "",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "Z492MWW122S",
        "makerName": "LG",
        "makerCode": "13",
        "brandName": "오브제컬렉션",
        "brandCode": "13",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "23",
                "price": "5200000",
                "modelCode": "Z492MWW122S",
                "popularScore": "4.9",
                "catalogDescription": """
                    스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 2등급(24.01 기준)/
                    [디자인] 네이처(메탈)/화이트/
                    [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/
                    [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채/과일, 육류/생선(생동), 쌀/잡곡, 감자/고구마/유산균김치/유산균인디케이터/
                    [편의] 칸칸탈취/항균핸들/리니어인버터/구입김치스캔보관/WiFi/스퀘어핸들/크기(가로x세로x깊이): 835x1860x838mm
                """,
                "catalogOption": "",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "Z492SGS123S",
        "makerName": "LG",
        "makerCode": "13",
        "brandName": "오브제컬렉션",
        "brandCode": "13",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "22",
                "price": "5113460",
                "modelCode": "Z492SGS123S",
                "popularScore": "4.9",
                "catalogDescription": """
                    스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 3등급(24.01 기준)/
                    [디자인] 솔리드(스테인리스)/(상단) 솔리드그린 (중/하단) 솔리드실버/
                    [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/
                    [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채/과일, 육류/생선(생동), 쌀/잡곡, 열대과일, 감자/고구마/유산균김치/유산균인디케이터/
                    [편의] 탈취/항균핸들/칸칸전원/리니어인버터/구입김치스캔보관/WiFi/스퀘어핸들/크기(가로x세로x깊이): 835x1860x838mm
                """,
                "catalogOption": "",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "Z323MEF",
        "makerName": "LG",
        "makerCode": "13",
        "brandName": "오브제컬렉션",
        "brandCode": "13",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "21",
                "price": "928510",
                "modelCode": "Z323MEF",
                "popularScore": "4.9",
                "catalogDescription": """
                    스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/
                    [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/
                    [용기] 용기: 8개/용기용량: 81.6L/소형김치통/
                    [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/
                    [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm
                """,
                "catalogOption": "",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
    {
        "shopName": shop_name,
        "catalogName": "김치플러스 코타",
        "makerName": "삼성전자",
        "makerCode": "14",
        "brandName": "비스포크",
        "brandCode": "13",
        "catalogCreatedAt": "2025-03-23",
        "catalogModifiedAt": "2025-03-23",
        "catalogSeries": [
            {
                "catalogCode": "30",
                "price": "9500000",
                "modelCode": "RQ45A94Y1",
                "popularScore": "4.9",
                "catalogDescription": """
                    스탠드형/4도어(3룸)/용량: 490L/2022년형/에너지: 1등급(21.12 기준)/
                    [디자인] 코타(메탈)/(상단) 코타썬옐로우 (중/하단) 코타그리너리/
                    [용기] 용기: 9개/상칸: 3개/중칸: 4개/하칸: 2개/용기용량: 114.9L/안심김치통/
                    [냉각] 냉동겸용칸: 중칸, 하칸/독립냉각/메탈쿨링커튼/아삭모드/도어포켓/냉동/보관모드: 구입김치, 저염김치, 야채/과일, 육류/생선(생동), 쌀/잡곡, 열대과일, 감자/고구마, 와인, 상온/저온쿨링숙성/숙성시간표시/발효숙성모드: 물김치/무김치, 육류/
                    [편의] 칸칸탈취/오토클로징/칸칸전원/LED라이팅/인버터/WiFi/크기(가로x세로x깊이): 795x1853x794mm
                """,
                "catalogOption": "",
                "seriesCreatedAt": "2025-03-23",
                "seriesModifiedAt": "2025-03-23"
            }
        ]
    },
]

# # 데이터 인덱싱
for catalog_data in catalog_data_list:
    # 데이터 인덱싱
    response = client.index(index='catalog', body=catalog_data)

    # print(f"리뷰: {restaurant_data['review']}  인덱싱 임베딩 벡터: {restaurant_data['review_vector']}")
    if response['result'] == 'created':
        print(f"'{catalog_data['shopName']}' 데이터가 인덱싱되었습니다.")
    else:
        print(f"'{catalog_data['shopName']}' 데이터 인덱싱 실패:", response)
