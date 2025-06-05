def vector_search_query(query_text, model_id):
    return {
        "query": {
            "neural": {
                "review_vector": {
                    "query_text": query_text,
                    "model_id": model_id,
                    "k": 100
                }
            }
        },
        "_source": {"excludes": ["review_vector"]}
    }


def semantic_search_query(query_text):
    return {
        "query": {
            "match": {"review": query_text}
        },
        "_source": {"excludes": ["review_vector"]}
    }


def hybrid_search_query(query_text, model_id):
    return {
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {
                                "neural": {
                                    "review_vector": {
                                        "query_text": query_text,
                                        "model_id": model_id,
                                        "k": 100
                                    }
                                }
                            },
                            "script": {"source": "_score * 1.5"}
                        }
                    },
                    {
                        "script_score": {
                            "query": {"match": {"review": query_text}},
                            "script": {"source": "_score * 1.7"}
                        }
                    }
                ]
            }
        },
        "_source": {"excludes": ["review_vector"]}
    }


# 기존 호환성을 위해 alias 제공
def query_maker(query, model_id):
    return hybrid_search_query(query, model_id)
