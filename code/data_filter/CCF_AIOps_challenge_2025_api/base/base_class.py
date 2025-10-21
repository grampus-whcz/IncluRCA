from abc import ABC
# from data_filter.CCF_AIOps_challenge_2025_api.config.dataset_config import DataConfig
from data_filter.CCF_AIOps_challenge_2025_api.config.dataset_config_phasetwo import DataConfig
from shared_util.logger import Logger
from datetime import datetime, timedelta


class BaseClass(ABC):
    def __init__(self):
        self.config = DataConfig()
        self.logger = Logger(self.config.param_dict['logging']['level']).logger

        self.all_entity_list = []
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['node_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['service_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['tidb_order'])
        self.all_entity_list.extend(self.config.data_dict['setting']['metric']['pod_order'])
        

    @staticmethod
    def rename_pod2service(pod_name):
        return pod_name.replace('-0', '').replace('-1', '').replace('-2', '')

    @staticmethod
    def rename_service2pod(service_name):
        if service_name == "redis-cart":
            return [f'{service_name}-0']
        else:
            return [f'{service_name}-0', f'{service_name}-1', f'{service_name}-2']
        
    @staticmethod
    def rename_service2api(service_name):
        if service_name == "adservice":
            return ["hipstershop.AdService/GetAds"]
        elif service_name == "cartservice":
            return ["hipstershop.CartService/AddItem", "hipstershop.CartService/GetCart", "hipstershop.CartService/EmptyCart",
                    "POST /hipstershop.CartService/AddItem", "POST /hipstershop.CartService/EmptyCart", "POST /hipstershop.CartService/GetCart"]
        elif service_name == "redis-cart":
            return ["HGET", "HMSET", "SET", "GET"]
        elif service_name == "checkoutservice":
            return ["hipstershop.CheckoutService/PlaceOrder"]
        elif service_name == "currencyservice":
            return ["hipstershop.CurrencyService/Convert", "hipstershop.CurrencyService/GetSupportedCurrencies"]
        elif service_name == "emailservice":
            return ["hipstershop.EmailService/SendOrderConfirmation", "/hipstershop.EmailService/SendOrderConfirmation"]
        elif service_name == "paymentservice":
            return ["hipstershop.PaymentService/Charge"]
        elif service_name == "productcatalogservice":
            return ["hipstershop.ProductCatalogService/GetProduct", "hipstershop.ProductCatalogService/ListProducts", "/hipstershop.ProductCatalogService/ListProducts"]
        elif service_name == "recommendationservice":
            return ["hipstershop.RecommendationService/ListRecommendations", "/hipstershop.RecommendationService/ListRecommendations"]
        elif service_name == "shippingservice":
            return ["hipstershop.ShippingService/GetQuote", "hipstershop.ShippingService/ShipOrder"]
        elif service_name == "frontend":
            return ["hipstershop.Frontend/Recv."]
        else:
            return None

