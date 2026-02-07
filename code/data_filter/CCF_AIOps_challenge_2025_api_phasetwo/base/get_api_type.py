import pyarrow.parquet as pq
import pyarrow as pa
import os

def get_unique_operation_names_by_filename(file_path):
    try:
        table = pq.read_table(file_path)
        batches = table.to_batches(max_chunksize=65536)
        for batch in batches:
            for record in batch.to_pylist():
                operation_name = record.get("operationName")
                if operation_name is not None and (operation_name == "SET" or operation_name == "GET"):
                    print(record)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    


def get_unique_operation_names(parquet_dir):
    operation_names = set()

    for root, dirs, files in os.walk(parquet_dir):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                try:
                    table = pq.read_table(file_path)
                    batches = table.to_batches(max_chunksize=65536)
                    for batch in batches:
                        for record in batch.to_pylist():
                            operation_name = record.get("operationName")
                            if operation_name is not None:
                                operation_names.add(operation_name)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            print(operation_names)

    return list(operation_names)  # 或者直接返回 set

def union_all():
    a6 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'SET', 'hipstershop.CurrencyService/Convert', 'GET', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.PaymentService/Charge', 'hipstershop.CartService/AddItem', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a7 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'SET', 'hipstershop.CurrencyService/Convert', 'GET', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.PaymentService/Charge', 'hipstershop.CartService/AddItem', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a8 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'hipstershop.CurrencyService/Convert', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a9 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'hipstershop.CurrencyService/Convert', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a10 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'SET', 'hipstershop.CurrencyService/Convert', 'GET', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a11 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'SET', 'hipstershop.CurrencyService/Convert', 'GET', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a12 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'SET', 'hipstershop.CurrencyService/Convert', 'GET', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.PaymentService/Charge', 'hipstershop.CartService/AddItem', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a13 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'hipstershop.CurrencyService/Convert', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    a14 = set(['hipstershop.CartService/EmptyCart', 'hipstershop.ShippingService/GetQuote', 'POST /hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/ListProducts', 'hipstershop.AdService/GetAds', 'hipstershop.CurrencyService/Convert', '/hipstershop.ProductCatalogService/ListProducts', 'hipstershop.EmailService/SendOrderConfirmation', 'hipstershop.CheckoutService/PlaceOrder', 'HMSET', 'hipstershop.CurrencyService/GetSupportedCurrencies', '/hipstershop.RecommendationService/ListRecommendations', 'hipstershop.RecommendationService/ListRecommendations', 'POST /hipstershop.CartService/AddItem', '/hipstershop.EmailService/SendOrderConfirmation', 'HGET', 'hipstershop.CartService/GetCart', 'hipstershop.ProductCatalogService/GetProduct', 'hipstershop.ShippingService/ShipOrder', 'hipstershop.CartService/AddItem', 'hipstershop.PaymentService/Charge', 'POST /hipstershop.CartService/EmptyCart', 'hipstershop.Frontend/Recv.'])
    union_them_all = a6 | a7 | a8 | a9 | a10 | a11 | a12 | a13 | a14
    print(union_them_all) 



if __name__ == '__main__':
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/trace-parquet")
    # print("##############################6##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-07/cloudbed/trace-parquet")
    # print("##############################7##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-08/cloudbed/trace-parquet")
    # print("##############################8##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-09/cloudbed/trace-parquet")
    # print("##############################9##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-10/cloudbed/trace-parquet")
    # print("##############################10##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-11/cloudbed/trace-parquet")
    # print("##############################11##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-12/cloudbed/trace-parquet")
    # print("##############################12##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-13/cloudbed/trace-parquet")
    # print("##############################13##################################")
    # print(aa)
    # aa = get_unique_operation_names("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-14/cloudbed/trace-parquet")
    # print("##############################14##################################")
    # print(aa)
    # union_all()
    
    
    # {'POST /hipstershop.CartService/AddItem', 
    #  'SET', 
    #  'hipstershop.RecommendationService/ListRecommendations', 
    #  'POST /hipstershop.CartService/EmptyCart', 
    #  'hipstershop.CartService/GetCart', 
    #  'GET', 
    #  'hipstershop.Frontend/Recv.', 
    #  'hipstershop.CurrencyService/Convert', 
    #  'hipstershop.CurrencyService/GetSupportedCurrencies', 
    #  'hipstershop.AdService/GetAds', 
    #  'HGET', 
    #  'POST /hipstershop.CartService/GetCart', 
    #  'hipstershop.CartService/AddItem', 
    #  'hipstershop.ShippingService/GetQuote', 
    #  'hipstershop.CartService/EmptyCart', 
    #  'HMSET', 
    #  '/hipstershop.EmailService/SendOrderConfirmation', 
    #  'hipstershop.ShippingService/ShipOrder', 
    #  'hipstershop.CheckoutService/PlaceOrder', 
    #  'hipstershop.PaymentService/Charge', 
    #  'hipstershop.EmailService/SendOrderConfirmation', 
    #  'hipstershop.ProductCatalogService/GetProduct', 
    #  '/hipstershop.ProductCatalogService/ListProducts', 
    #  '/hipstershop.RecommendationService/ListRecommendations', 
    #  'hipstershop.ProductCatalogService/ListProducts'}
    
    get_unique_operation_names_by_filename("/root/shared-nvme/data_set/2025_CCF_aiops-live-benchmark/phaseone/2025-06-06/cloudbed/trace-parquet/trace_jaeger-span_2025-06-06_02-00-00.parquet")
    
    # {'traceID': '9c5d4f9a5f0cbe542fd2ec2150ec6fd3', 'spanID': 'e345aa813aefe51d', 'flags': None, 'operationName': 'SET', 'references': [{'refType': 'CHILD_OF', 'spanID': 'fd974c42ff10178a', 'traceID': '9c5d4f9a5f0cbe542fd2ec2150ec6fd3'}], 'startTime': 1749148454105798, 'startTimeMillis': 1749148454105, 'duration': 658288, 'tags': [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.StackExchangeRedis'}, {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.10'}, {'key': 'db.system', 'type': 'string', 'value': 'redis'}, {'key': 'db.redis.flags', 'type': 'string', 'value': 'DemandMaster'}, {'key': 'db.statement', 'type': 'string', 'value': 'SET'}, {'key': 'net.peer.name', 'type': 'string', 'value': 'redis-cart'}, {'key': 'net.peer.port', 'type': 'int64', 'value': '6379'}, {'key': 'db.redis.database_index', 'type': 'int64', 'value': '0'}, {'key': 'peer.service', 'type': 'string', 'value': 'redis-cart:6379'}, {'key': 'span.kind', 'type': 'string', 'value': 'client'}, {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}], 'logs': [{'fields': [{'key': 'event', 'type': 'string', 'value': 'Enqueued'}], 'timestamp': 1749148454168853}, {'fields': [{'key': 'event', 'type': 'string', 'value': 'Sent'}], 'timestamp': 1749148454169698}, {'fields': [{'key': 'event', 'type': 'string', 'value': 'ResponseReceived'}], 'timestamp': 1749148454763580}], 'process': {'serviceName': 'redis', 'tags': [{'key': 'exporter', 'type': 'string', 'value': 'jaeger'}, {'key': 'float', 'type': 'float64', 'value': '312.23'}, {'key': 'ip', 'type': 'string', 'value': '10.233.89.126'}, {'key': 'podName', 'type': 'string', 'value': 'cartservice-1'}, {'key': 'nodeName', 'type': 'string', 'value': 'aiops-k8s-08'}, {'key': 'namespace', 'type': 'string', 'value': 'hipstershop'}, {'key': 'service.instance.id', 'type': 'string', 'value': '31bc6d51-aa06-4db8-a12e-ed4e2f8f8624'}, {'key': 'telemetry.sdk.name', 'type': 'string', 'value': 'opentelemetry'}, {'key': 'telemetry.sdk.language', 'type': 'string', 'value': 'dotnet'}, {'key': 'telemetry.sdk.version', 'type': 'string', 'value': '1.6.0'}]}}
    # {'traceID': '9021c85c389fa0766b0666146791b903', 'spanID': 'd0152b516654edf7', 'flags': None, 'operationName': 'GET', 'references': [{'refType': 'CHILD_OF', 'spanID': '57475ca85f68b6b5', 'traceID': '9021c85c389fa0766b0666146791b903'}], 'startTime': 1749148460895982, 'startTimeMillis': 1749148460895, 'duration': 10596, 'tags': [{'key': 'otel.library.name', 'type': 'string', 'value': 'OpenTelemetry.Instrumentation.StackExchangeRedis'}, {'key': 'otel.library.version', 'type': 'string', 'value': '1.0.0.10'}, {'key': 'db.system', 'type': 'string', 'value': 'redis'}, {'key': 'db.redis.flags', 'type': 'string', 'value': 'None'}, {'key': 'db.statement', 'type': 'string', 'value': 'GET'}, {'key': 'net.peer.name', 'type': 'string', 'value': 'redis-cart'}, {'key': 'net.peer.port', 'type': 'int64', 'value': '6379'}, {'key': 'db.redis.database_index', 'type': 'int64', 'value': '0'}, {'key': 'peer.service', 'type': 'string', 'value': 'redis-cart:6379'}, {'key': 'span.kind', 'type': 'string', 'value': 'client'}, {'key': 'internal.span.format', 'type': 'string', 'value': 'otlp'}], 'logs': [{'fields': [{'key': 'event', 'type': 'string', 'value': 'Enqueued'}], 'timestamp': 1749148460901048}, {'fields': [{'key': 'event', 'type': 'string', 'value': 'Sent'}], 'timestamp': 1749148460901543}, {'fields': [{'key': 'event', 'type': 'string', 'value': 'ResponseReceived'}], 'timestamp': 1749148460906404}], 'process': {'serviceName': 'redis', 'tags': [{'key': 'exporter', 'type': 'string', 'value': 'jaeger'}, {'key': 'float', 'type': 'float64', 'value': '312.23'}, {'key': 'ip', 'type': 'string', 'value': '10.233.81.24'}, {'key': 'podName', 'type': 'string', 'value': 'cartservice-2'}, {'key': 'nodeName', 'type': 'string', 'value': 'aiops-k8s-03'}, {'key': 'namespace', 'type': 'string', 'value': 'hipstershop'}, {'key': 'service.instance.id', 'type': 'string', 'value': 'de05632e-1281-4bfd-adc2-f40239001eea'}, {'key': 'telemetry.sdk.name', 'type': 'string', 'value': 'opentelemetry'}, {'key': 'telemetry.sdk.language', 'type': 'string', 'value': 'dotnet'}, {'key': 'telemetry.sdk.version', 'type': 'string', 'value': '1.6.0'}]}}