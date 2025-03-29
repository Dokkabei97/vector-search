def query_maker(query, model_id):
    query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "script_score": {
                            "query": {
                                "neural": {
                                    "review_vector": {
                                        "query_text": query,
                                        "model_id": model_id,
                                        "k": 100
                                    }
                                }
                            },
                            "script": {
                                "source": "_score * 1.5"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "query": {
                                "match": {
                                    "review": query
                                }
                            },
                            "script": {
                                "source": "_score * 1.7"
                            }
                        }
                    }
                ]
            }
        },
        "_source": {
            "excludes": ["review_vector"]
        }
    }