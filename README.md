# vector-search

### Description
Opensearch를 이용한 벡터 검색 및 시멘틱 서치

```json
PUT _cluster/settings
{
  "persistent": {
    "plugins.ml_commons.allow_registering_model_via_url": "true",
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99"
  }
}
```

```json
# access_mode 옵션은 Security 플러그인이 활성화 되어야 함
POST /_plugins/_ml/model_groups/_register
{
  "name": "kbert_group",
  "description": "korean"
}
```

```shell
sha -a 256 kbert.zip
```

```json
POST /_plugins/_ml/models/_register
{
    "name": "kbert",
    "version": "1.0.0",
    "description": "korean bert model",
    "model_format": "ONNX",
    "model_group_id": "모델 그룹 ID",
    "model_content_hash_value": "모델 해시값",
    "model_config": {
        "model_type": "bert",
        "embedding_dimension": 384,
        "framework_type": "sentence_transformers",
       "all_config": "{\"_name_or_path\":\"nreimers/MiniLM-L6-H384-uncased\",\"architectures\":[\"BertModel\"],\"attention_probs_dropout_prob\":0.1,\"gradient_checkpointing\":false,\"hidden_act\":\"gelu\",\"hidden_dropout_prob\":0.1,\"hidden_size\":384,\"initializer_range\":0.02,\"intermediate_size\":1536,\"layer_norm_eps\":1e-12,\"max_position_embeddings\":512,\"model_type\":\"bert\",\"num_attention_heads\":12,\"num_hidden_layers\":6,\"pad_token_id\":0,\"position_embedding_type\":\"absolute\",\"transformers_version\":\"4.8.2\",\"type_vocab_size\":2,\"use_cache\":true,\"vocab_size\":30522}"
    },
    "url": "URL"
}
```

```http request
GET /_plugins/_ml/tasks/태스크 ID

POST /_plugins/_ml/models/모델 ID/_deploy

GET /_plugins/_ml/tasks/태스크 ID
```

Ingest Pipeline Setting
```json
PUT _ingest/pipeline/bert-pipeline
{
  "processors": [
    {
      "text_embedding": {
        "model_id": "모델 ID",
        "field_map": {
          "필드": "벡터필드"
        }
      }
    }
  ]
}
```

Index Mapping
```json
PUT catalog
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": true,
        "default_pipeline": "bert-pipeline"
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
              "store": true,
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
                    "space_type": "cosinesimil",  
                    "engine": "lucene",
                    "parameters": {}
                }
            }
        }
    }
}
```

Bulk API
```json
POST _bulk
{"index":{"_index":"catalog","_id":"1"}}
{"shopName":"abc","catalogName":"신라면 120g","makerName":"농심","makerCode":"10","brandName":"신라면","brandCode":"10","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"1","price":"700","modelCode":"1","popularScore":"4.5","catalogDescription":"봉지라면","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"2","price":"7000","modelCode":"2","popularScore":"4.9","catalogDescription":"봉지라면","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"3","price":"21000","modelCode":"3","popularScore":"4.2","catalogDescription":"봉지라면","catalogOption":"30개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"4","price":"35000","modelCode":"4","popularScore":"4.7","catalogDescription":"봉지라면","catalogOption":"50개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"농심 신라면 120g 봉지라면 1개 10개 30개 50개 700원 7000원 21000원 35000원"}
{"index":{"_index":"catalog","_id":"2"}}
{"shopName":"abc","catalogName":"신라면 65g","makerName":"농심","makerCode":"10","brandName":"신라면","brandCode":"10","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"5","price":"800","modelCode":"5","popularScore":"4.9","catalogDescription":"컵라면","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"6","price":"7600","modelCode":"6","popularScore":"4.9","catalogDescription":"컵라면","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"농심 신라면 65g 컵라면 1개 10개 800원 7600원"}
{"index":{"_index":"catalog","_id":"3"}}
{"shopName":"abc","catalogName":"신라면 블랙 120g","makerName":"농심","makerCode":"10","brandName":"신라면","brandCode":"10","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"7","price":"1000","modelCode":"7","popularScore":"4.9","catalogDescription":"봉지라면","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"8","price":"10000","modelCode":"8","popularScore":"4.9","catalogDescription":"봉지라면","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"농심 신라면 블랙 120g 봉지라면 1개 10개 1000원 10000원"}
{"index":{"_index":"catalog","_id":"4"}}
{"shopName":"abc","catalogName":"진라면 매운맛 120g","makerName":"오뚜기","makerCode":"11","brandName":"진라면","brandCode":"11","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"9","price":"720","modelCode":"9","popularScore":"4.9","catalogDescription":"봉지라면 / 매운맛","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"10","price":"6900","modelCode":"10","popularScore":"4.6","catalogDescription":"봉지라면 / 매운맛","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"오뚜기 진라면 매운맛 120g 봉지라면 / 매운맛 1개 10개 720원 6900원"}
{"index":{"_index":"catalog","_id":"5"}}
{"shopName":"abc","catalogName":"진라면 순한맛 120g","makerName":"오뚜기","makerCode":"11","brandName":"진라면","brandCode":"11","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"11","price":"720","modelCode":"11","popularScore":"4.9","catalogDescription":"봉지라면 / 순한맛","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"12","price":"6900","modelCode":"12","popularScore":"4.6","catalogDescription":"봉지라면 / 순한맛","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"오뚜기 진라면 순한맛 120g 봉지라면 / 순한맛 1개 10개 720원 6900원"}
{"index":{"_index":"catalog","_id":"6"}}
{"shopName":"abc","catalogName":"진라면 순한맛 70g","makerName":"오뚜기","makerCode":"11","brandName":"진라면","brandCode":"11","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"13","price":"900","modelCode":"13","popularScore":"4.9","catalogDescription":"컵라면 / 순한맛","catalogOption":"1개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"},{"catalogCode":"14","price":"8800","modelCode":"14","popularScore":"4.6","catalogDescription":"컵라면 / 순한맛","catalogOption":"10개","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"오뚜기 신라면 순한맛 70g 컵라면 / 순한맛 1개 10개 900원 8800원"}
{"index":{"_index":"catalog","_id":"7"}}
{"shopName":"abc","catalogName":"Z339NEEF11","makerName":"LG","makerCode":"13","brandName":"오브제컬렉션","brandCode":"13","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"20","price":"1228770","modelCode":"Z339NEEF11","popularScore":"4.9","catalogDescription":"스탠드형/3도어/용량: 327L/2025년형/에너지: 1등급(24.09 기준)/ [디자인] 네이처(메탈)/베이지/ [용기] 용기: 4개/상칸: 미포함/중칸: 2개/하칸: 2개/용기용량: 56.8L/반투명김치통/ [냉각] 냉동겸용칸: 상칸(전체)/순환냉각/쿨링케어/냉기지킴가드/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡/유산균김치/유산균인디케이터/","catalogOption":"","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"LG 오브제컬렉션 Z339NEEF11 스탠드형/3도어/용량: 327L/2025년형/에너지: 1등급(24.09 기준)/ [디자인] 네이처(메탈)/베이지/ [용기] 용기: 4개/상칸: 미포함/중칸: 2개/하칸: 2개/용기용량: 56.8L/반투명김치통/ [냉각] 냉동겸용칸: 상칸(전체)/순환냉각/쿨링케어/냉기지킴가드/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡/유산균김치/유산균인디케이터/ 1228770원"}
{"index":{"_index":"catalog","_id":"8"}}
{"shopName":"abc","catalogName":"Z492MWW122S","makerName":"LG","makerCode":"13","brandName":"오브제컬렉션","brandCode":"13","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"23","price":"5200000","modelCode":"Z492MWW122S","popularScore":"4.9","catalogDescription":"스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 2등급(24.01 기준)/ [디자인] 네이처(메탈)/화이트/ [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/ [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 감자, 고구마/유산균김치/유산균인디케이터/","catalogOption":"","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"LG 오브제컬렉션 Z492MWW122S 스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 2등급(24.01 기준)/ [디자인] 네이처(메탈)/화이트/ [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/ [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 감자, 고구마/유산균김치/유산균인디케이터/ 5200000원"}
{"index":{"_index":"catalog","_id":"9"}}
{"shopName":"abc","catalogName":"Z492SGS123S","makerName":"LG","makerCode":"13","brandName":"오브제컬렉션","brandCode":"13","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"22","price":"5113460","modelCode":"Z492SGS123S","popularScore":"4.9","catalogDescription":"스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 3등급(24.01 기준)/ [디자인] 솔리드(스테인리스)/(상단) 솔리드그린 (중/하단) 솔리드실버/ [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/ [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 열대과일, 감자, 고구마/유산균김치/유산균인디케이터/","catalogOption":"","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"LG 오브제컬렉션 Z492SGS123S 스탠드형/4도어(4룸)/용량: 491L/2023년형/에너지: 3등급(24.01 기준)/ [디자인] 솔리드(스테인리스)/(상단) 솔리드그린 (중/하단) 솔리드실버/ [용기] 용기: 14개/상칸: 6개/중칸: 4개/하칸: 4개/투명김치통/반투명김치통/ [냉각] 냉동겸용칸: 상칸(일부)/순환냉각/쿨링케어/유산균가드/냉기지킴가드/신선야채실/도어포켓/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 열대과일, 감자, 고구마/유산균김치/유산균인디케이터/ 5113460원"}
{"index":{"_index":"catalog","_id":"10"}}
{"shopName":"abc","catalogName":"Z323MEF","makerName":"LG","makerCode":"13","brandName":"오브제컬렉션","brandCode":"13","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"21","price":"928510","modelCode":"Z323MEF","popularScore":"4.9","catalogDescription":"스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm","catalogOption":"","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"LG 오브제컬렉션 Z323MEF 스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm 928510원"}
{"index":{"_index":"catalog","_id":"11"}}
{"shopName":"abc","catalogName":"김치플러스 코타","makerName":"삼성전자","makerCode":"14","brandName":"비스포크","brandCode":"13","catalogCreatedAt":"2025-03-23","catalogModifiedAt":"2025-03-23","catalogSeries":[{"catalogCode":"30","price":"9500000","modelCode":"RQ45A94Y1","popularScore":"4.9","catalogDescription":"스탠드형/4도어(3룸)/용량: 490L/2022년형/에너지: 1등급(21.12 기준)/ [디자인] 코타(메탈)/(상단) 코타썬옐로우 (중/하단) 코타그리너리/ [용기] 용기: 9개/상칸: 3개/중칸: 4개/하칸: 2개/용기용량: 114.9L/안심김치통/ [냉각] 냉동겸용칸: 중칸, 하칸/독립냉각/메탈쿨링커튼/아삭모드/도어포켓/냉동/보관모드: 구입김치, 저염김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 열대과일, 감자, 고구마, 와인, 상온, 저온쿨링숙성/숙성시간표시/발효숙성모드: 물김치/무김치, 육류/ [편의] 칸칸탈취/오토클로징/칸칸전원/LED라이팅/인버터/WiFi/크기(가로x세로x깊이): 795x1853x794mm","catalogOption":"","seriesCreatedAt":"2025-03-23","seriesModifiedAt":"2025-03-23"}],"allDataSet":"삼성전자 김치플러스 코타 RQ45A94Y1 스탠드형/4도어(3룸)/용량: 490L/2022년형/에너지: 1등급(21.12 기준)/ [디자인] 코타(메탈)/(상단) 코타썬옐로우 (중/하단) 코타그리너리/ [용기] 용기: 9개/상칸: 3개/중칸: 4개/하칸: 2개/용기용량: 114.9L/안심김치통/ [냉각] 냉동겸용칸: 중칸, 하칸/독립냉각/메탈쿨링커튼/아삭모드/도어포켓/냉동/보관모드: 구입김치, 저염김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡, 열대과일, 감자, 고구마, 와인, 상온, 저온쿨링숙성/숙성시간표시/발효숙성모드: 물김치/무김치, 육류/ [편의] 칸칸탈취/오토클로징/칸칸전원/LED라이팅/인버터/WiFi/크기(가로x세로x깊이): 795x1853x794mm 9500000원"}
```

Query
```json
GET catalog/_search
{
  "track_total_hits": true, 
  "_source": {
    "excludes": "totalIndex_vector"
  },
  "query": {
    "hybrid": {
      "queries": [
        {
          "neural": {
            "totalIndex_vector": {
              "query_text": "나는 400L 정도의 김치냉장고를 원해",
              "model_id": "8lda4pUBtJNcBu1zpca8",
              "k": 100
            }
          }
        },
        {
          "bool": {
            "must": [
              {
                "match": {
                  "catalogDescription": "400L"
                }
              }
            ]
          }
        }
      ]
    }
  }
}
```

Response
```json
{
  "took": 31,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 11,
      "relation": "eq"
    },
    "max_score": 0.73952174,
    "hits": [
      {
        "_index": "catalog",
        "_id": "10",
        "_score": -9549512000,
        "_source": {
          "catalogName": "Z323MEF",
          "makerCode": "13",
          "brandName": "오브제컬렉션",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "LG 오브제컬렉션 Z323MEF 스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm 928510원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "928510",
              "modelCode": "Z323MEF",
              "popularScore": "4.9",
              "catalogCode": "21",
              "catalogOption": "",
              "catalogDescription": "스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "LG",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "13"
        }
      },
      {
        "_index": "catalog",
        "_id": "10",
        "_score": -4422440400,
        "_source": {
          "catalogName": "Z323MEF",
          "makerCode": "13",
          "brandName": "오브제컬렉션",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "LG 오브제컬렉션 Z323MEF 스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm 928510원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "928510",
              "modelCode": "Z323MEF",
              "popularScore": "4.9",
              "catalogCode": "21",
              "catalogOption": "",
              "catalogDescription": "스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "LG",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "13"
        }
      },
      {
        "_index": "catalog",
        "_id": "10",
        "_score": 0.73952174,
        "_source": {
          "catalogName": "Z323MEF",
          "makerCode": "13",
          "brandName": "오브제컬렉션",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "LG 오브제컬렉션 Z323MEF 스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm 928510원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "928510",
              "modelCode": "Z323MEF",
              "popularScore": "4.9",
              "catalogCode": "21",
              "catalogOption": "",
              "catalogDescription": "스탠드형/1도어/용량: 324L/2024년형/에너지: 1등급(24.02 기준)/ [디자인] 고내디자인: 화이트디자인/네이처(메탈)/네이처베이지/ [용기] 용기: 8개/용기용량: 81.6L/소형김치통/ [냉각] 멀티냉각/순환냉각/쿨링케어/냉기지킴가드/신선야채실/냉동/익힘/ [편의] 탈취/가변도어/인버터/구입김치스캔보관/WiFi/이지핸들/크기(가로x세로x깊이): 595x1860x707mm",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "LG",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "13"
        }
      },
      {
        "_index": "catalog",
        "_id": "2",
        "_score": 0.7378223,
        "_source": {
          "catalogName": "신라면 65g",
          "makerCode": "10",
          "brandName": "신라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "농심 신라면 65g 컵라면 1개 10개 800원 7600원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "800",
              "modelCode": "5",
              "popularScore": "4.9",
              "catalogCode": "5",
              "catalogOption": "1개",
              "catalogDescription": "컵라면",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "7600",
              "modelCode": "6",
              "popularScore": "4.9",
              "catalogCode": "6",
              "catalogOption": "10개",
              "catalogDescription": "컵라면",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "농심",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "10"
        }
      },
      {
        "_index": "catalog",
        "_id": "4",
        "_score": 0.7296756,
        "_source": {
          "catalogName": "진라면 매운맛 120g",
          "makerCode": "11",
          "brandName": "진라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "오뚜기 진라면 매운맛 120g 봉지라면 / 매운맛 1개 10개 720원 6900원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "720",
              "modelCode": "9",
              "popularScore": "4.9",
              "catalogCode": "9",
              "catalogOption": "1개",
              "catalogDescription": "봉지라면 / 매운맛",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "6900",
              "modelCode": "10",
              "popularScore": "4.6",
              "catalogCode": "10",
              "catalogOption": "10개",
              "catalogDescription": "봉지라면 / 매운맛",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "오뚜기",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "11"
        }
      },
      {
        "_index": "catalog",
        "_id": "7",
        "_score": 0.7292499,
        "_source": {
          "catalogName": "Z339NEEF11",
          "makerCode": "13",
          "brandName": "오브제컬렉션",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "LG 오브제컬렉션 Z339NEEF11 스탠드형/3도어/용량: 327L/2025년형/에너지: 1등급(24.09 기준)/ [디자인] 네이처(메탈)/베이지/ [용기] 용기: 4개/상칸: 미포함/중칸: 2개/하칸: 2개/용기용량: 56.8L/반투명김치통/ [냉각] 냉동겸용칸: 상칸(전체)/순환냉각/쿨링케어/냉기지킴가드/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡/유산균김치/유산균인디케이터/ 1228770원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "1228770",
              "modelCode": "Z339NEEF11",
              "popularScore": "4.9",
              "catalogCode": "20",
              "catalogOption": "",
              "catalogDescription": "스탠드형/3도어/용량: 327L/2025년형/에너지: 1등급(24.09 기준)/ [디자인] 네이처(메탈)/베이지/ [용기] 용기: 4개/상칸: 미포함/중칸: 2개/하칸: 2개/용기용량: 56.8L/반투명김치통/ [냉각] 냉동겸용칸: 상칸(전체)/순환냉각/쿨링케어/냉기지킴가드/냉동/맛지킴/익힘/오래보관/보관모드: 구입김치, 야채, 과일, 육류, 생선(생동), 쌀/잡곡/유산균김치/유산균인디케이터/",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "LG",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "13"
        }
      },
      {
        "_index": "catalog",
        "_id": "5",
        "_score": 0.7277683,
        "_source": {
          "catalogName": "진라면 순한맛 120g",
          "makerCode": "11",
          "brandName": "진라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "오뚜기 진라면 순한맛 120g 봉지라면 / 순한맛 1개 10개 720원 6900원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "720",
              "modelCode": "11",
              "popularScore": "4.9",
              "catalogCode": "11",
              "catalogOption": "1개",
              "catalogDescription": "봉지라면 / 순한맛",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "6900",
              "modelCode": "12",
              "popularScore": "4.6",
              "catalogCode": "12",
              "catalogOption": "10개",
              "catalogDescription": "봉지라면 / 순한맛",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "오뚜기",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "11"
        }
      },
      {
        "_index": "catalog",
        "_id": "3",
        "_score": 0.7121208,
        "_source": {
          "catalogName": "신라면 블랙 120g",
          "makerCode": "10",
          "brandName": "신라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "농심 신라면 블랙 120g 봉지라면 1개 10개 1000원 10000원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "1000",
              "modelCode": "7",
              "popularScore": "4.9",
              "catalogCode": "7",
              "catalogOption": "1개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "10000",
              "modelCode": "8",
              "popularScore": "4.9",
              "catalogCode": "8",
              "catalogOption": "10개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "농심",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "10"
        }
      },
      {
        "_index": "catalog",
        "_id": "1",
        "_score": 0.7113477,
        "_source": {
          "catalogName": "신라면 120g",
          "makerCode": "10",
          "brandName": "신라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "농심 신라면 120g 봉지라면 1개 10개 30개 50개 700원 7000원 21000원 35000원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "700",
              "modelCode": "1",
              "popularScore": "4.5",
              "catalogCode": "1",
              "catalogOption": "1개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "7000",
              "modelCode": "2",
              "popularScore": "4.9",
              "catalogCode": "2",
              "catalogOption": "10개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "21000",
              "modelCode": "3",
              "popularScore": "4.2",
              "catalogCode": "3",
              "catalogOption": "30개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "35000",
              "modelCode": "4",
              "popularScore": "4.7",
              "catalogCode": "4",
              "catalogOption": "50개",
              "catalogDescription": "봉지라면",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "농심",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "10"
        }
      },
      {
        "_index": "catalog",
        "_id": "6",
        "_score": 0.706195,
        "_source": {
          "catalogName": "진라면 순한맛 70g",
          "makerCode": "11",
          "brandName": "진라면",
          "catalogCreatedAt": "2025-03-23",
          "allDataSet": "오뚜기 신라면 순한맛 70g 컵라면 / 순한맛 1개 10개 900원 8800원",
          "shopName": "abc",
          "catalogSeries": [
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "900",
              "modelCode": "13",
              "popularScore": "4.9",
              "catalogCode": "13",
              "catalogOption": "1개",
              "catalogDescription": "컵라면 / 순한맛",
              "seriesCreatedAt": "2025-03-23"
            },
            {
              "seriesModifiedAt": "2025-03-23",
              "price": "8800",
              "modelCode": "14",
              "popularScore": "4.6",
              "catalogCode": "14",
              "catalogOption": "10개",
              "catalogDescription": "컵라면 / 순한맛",
              "seriesCreatedAt": "2025-03-23"
            }
          ],
          "makerName": "오뚜기",
          "catalogModifiedAt": "2025-03-23",
          "brandCode": "11"
        }
      }
    ]
  }
}
```