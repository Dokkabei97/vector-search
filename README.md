# vector-search

### Description
Opensearch를 이용한 벡터 검색 및 시멘틱 서치

```http request
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

```http request
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

```http request
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
    "url": "http://minio:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL21vZGVsL2tiZXJ0LnppcD9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUdQOURGR0daSDIxUURXNkhYWUFWJTJGMjAyNDEwMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI3VDEwMTIwNFomWC1BbXotRXhwaXJlcz00MzE5OSZYLUFtei1TZWN1cml0eS1Ub2tlbj1leUpoYkdjaU9pSklVelV4TWlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKaFkyTmxjM05MWlhraU9pSkhVRGxFUmtkSFdrZ3lNVkZFVnpaSVdGbEJWaUlzSW1WNGNDSTZNVGN6TURBMk1ERXlNQ3dpY0dGeVpXNTBJam9pYldsdWFXOWtiMk5yWlhJaWZRLjJjZmQ1NTVmTmNfUHV4ZXhpcjlsSEh4ZUxoY0ttUE52NVVaQVZqV083YkZXRTFPTjNxdUxiVzZrNzRCX3B2ak5hNVJjWmYzbHByVmlRT0hQZUZySDZBJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9YmEyMmE0N2Q3NTA1YzQ3YjlmYzY2N2QyNzE4NmNmYTEzYmVjMmIzYjRkM2QyY2I2YmIzZGQ0ZjUwYTk2MzQ3Yg"
}
```

```http request
GET /_plugins/_ml/tasks/태스크 ID

POST /_plugins/_ml/models/모델 ID/_deploy

GET /_plugins/_ml/tasks/태스크 ID
```

```http request
PUT _ingest/pipeline/bert-pipeline
{
  "processors": [
    {
      "text_embedding": {
        "model_id": "모델 ID",
        "field_map": {
          "review": "review_vector"
        }
      }
    }
  ]
}
```

```http request
PUT restaurant_recommendation
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
                    "space_type": "cosinesimil",  
                    "engine": "lucene",  
                    "parameters": {}
                }
            }
        }
    }
}
```