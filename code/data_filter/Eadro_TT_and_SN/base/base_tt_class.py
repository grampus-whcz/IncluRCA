from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass


class BaseTTClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'TT'
        self.fault_interval = self.config.param_dict['fault_interval']['TT']
        self.sample_granularity = self.config.param_dict['sample_granularity']['TT']
        temp_service_list = [
            'assurance', 'auth', 'basic', 'cancel', 'config', 'contacts',
            'food-map', 'food', 'inside-payment', 'notification', 'order-other',
            'order', 'payment', 'preserve', 'price', 'route-plan', 'route',
            'seat', 'security', 'station', 'ticketinfo', 'train', 'travel-plan',
            'travel', 'travel2', 'user', 'verification-code'
        ]
        valid_network_service_list = [
            'assurance', 'auth', 'basic', 'contacts', 'order-other', 'order',
            'preserve', 'route', 'security', 'station', 'ticketinfo', 'travel', 'verification-code'
        ]
        self.all_entity_list = ['ts-{}-service'.format(api) for api in temp_service_list]
        self.valid_network_entity_list = ['ts-{}-service'.format(api) for api in valid_network_service_list]
        ent_edge_info = {
            "preserve": ["seat", "security", "food", "order", "ticketinfo",
                         "travel", "contacts", "notification", "user", "station"],
            "seat": ["order", "config", "travel"],
            "cancel": ["inside-payment", "order-other", "order"],
            "security": ["security", "order-other", "order"],
            "food": ["travel", "food-map", "station"],
            "travel": ["order", "ticketinfo", "train", "route"],
            "inside-payment": ["payment", "order"],
            "ticketinfo": ["basic"],
            "basic": ["route", "price", "train", "station"],
            "order-other": ["station"],
            "order": ["station", "assurance"],
            "auth": ["verification-code"]
        }
        self.ent_edge_index_list = [[], []]
        for i in range(len(self.ent_edge_index_list)):
            self.ent_edge_index_list[0].append(i)
            self.ent_edge_index_list[1].append(i)
        for start_edge, end_edge_list in ent_edge_info.items():
            for end_edge in end_edge_list:
                self.ent_edge_index_list[0].append(self.all_entity_list.index(f'ts-{start_edge}-service'))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(f'ts-{end_edge}-service'))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(f'ts-{start_edge}-service'))
                self.ent_edge_index_list[0].append(self.all_entity_list.index(f'ts-{end_edge}-service'))
                
        self.all_api_list = [
            'ts-assurance-service:getAllAssuranceType',
            'ts-auth-service:GET',
            'ts-auth-service:getToken',
            'ts-basic-service:GET',
            'ts-basic-service:queryForStationId',
            'ts-basic-service:queryForTravel',
            'ts-config-service:retrieve',
            'ts-contacts-service:findContactsByAccountId',
            'ts-contacts-service:getContactsByContactsId',
            'ts-food-map-service:getTrainFoodOfTrip',
            'ts-food-service:GET',
            'ts-order-other-service:securityInfoCheck',
            'ts-order-service:POST',
            'ts-order-service:calculateSoldTicket',
            'ts-order-service:getTicketListByDateAndTripId',
            'ts-order-service:queryOrdersForRefresh',
            'ts-order-service:securityInfoCheck',
            'ts-preserve-service:GET',
            'ts-preserve-service:preserve',
            'ts-price-service:query',
            'ts-route-service:queryById',
            'ts-seat-service:GET',
            'ts-seat-service:POST',
            'ts-seat-service:getLeftTicketOfInterval',
            'ts-security-service:GET',
            'ts-security-service:check',
            'ts-station-service:queryForNameBatch',
            'ts-station-service:queryForStationId',
            'ts-ticketinfo-service:GET',
            'ts-ticketinfo-service:POST',
            'ts-ticketinfo-service:queryForStationId',
            'ts-ticketinfo-service:queryForTravel',
            'ts-train-service:retrieve',
            'ts-travel-service:GET',
            'ts-travel-service:POST',
            'ts-travel-service:queryInfo',
            'ts-verification-code-service:imageCode',
            'ts-verification-code-service:verifyCode'
        ]
