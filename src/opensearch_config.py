from opensearchpy import OpenSearch


def setup_opensearch_client():
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        # 실제 환경에서는 아래 주석을 해제하고 인증 정보를 설정하세요.
        # http_auth=('username', 'password'),
        # use_ssl=True,
        # verify_certs=True,
        # ssl_show_warn=False
    )
    return client