from data_filter.Eadro_TT_and_SN.base.base_class import BaseClass


class BaseSNClass(BaseClass):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'SN'
        self.fault_interval = self.config.param_dict['fault_interval']['SN']
        self.sample_granularity = self.config.param_dict['sample_granularity']['SN']
        self.all_entity_list = [
            'social-graph-service', 'compose-post-service', 'post-storage-service',
            'user-timeline-service', 'url-shorten-service', 'user-service',
            'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
            'home-timeline-service', "nginx-web-server"
        ]
        ent_edge_info = {
            "compose-post-service": [
                "home-timeline-service", "media-service", "post-storage-service",
                "text-service", "unique-id-service", "user-service", "user-timeline-service"
            ],
            "home-timeline-service": [
                "post-storage-service", "social-graph-service"
            ],
            "social-graph-service": ["user-service"],
            "text-service": ["url-shorten-service", "user-mention-service"],
            "nginx-web-server": [
                "compose-post-service", "home-timeline-service",
                "social-graph-service", "user-service"
            ]
        }
        self.valid_network_entity_list = [
            'social-graph-service', 'compose-post-service', 'post-storage-service',
            'user-timeline-service', 'url-shorten-service', 'user-service',
            'media-service', 'text-service', 'unique-id-service', 'user-mention-service',
            'home-timeline-service', "nginx-web-server"
        ]
        self.ent_edge_index_list = [[], []]
        for i in range(len(self.all_entity_list)):
            self.ent_edge_index_list[0].append(i)
            self.ent_edge_index_list[1].append(i)
        for start_edge, end_edge_list in ent_edge_info.items():
            for end_edge in end_edge_list:
                self.ent_edge_index_list[0].append(self.all_entity_list.index(start_edge))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(end_edge))
                self.ent_edge_index_list[1].append(self.all_entity_list.index(start_edge))
                self.ent_edge_index_list[0].append(self.all_entity_list.index(end_edge))
                
        self.all_api_list = [
            'compose-post-service:compose_creator_client',
            'compose-post-service:compose_media_client',
            'compose-post-service:compose_post_server',
            'compose-post-service:compose_text_client',
            'compose-post-service:compose_unique_id_client',
            'compose-post-service:store_post_client',
            'compose-post-service:write_home_timeline_client',
            'compose-post-service:write_user_timeline_client',
            'home-timeline-service:get_followers_client',
            'home-timeline-service:read_home_timeline_redis_find_client',
            'home-timeline-service:read_home_timeline_server',
            'home-timeline-service:write_home_timeline_server',
            'media-service:compose_media_server',
            'nginx-web-server:/api/home-timeline/read',
            'nginx-web-server:/api/post/compose',
            'nginx-web-server:/api/user/follow',
            'nginx-web-server:/api/user/get_followee',
            'nginx-web-server:/api/user/get_follower',
            'nginx-web-server:/api/user/login',
            'nginx-web-server:/api/user/register',
            'nginx-web-server:/api/user/unfollow',
            'nginx-web-server:Follow',
            'nginx-web-server:GetFollowee',
            'nginx-web-server:GetFollower',
            'nginx-web-server:Login',
            'nginx-web-server:RegisterUser',
            'nginx-web-server:Unfollow',
            'nginx-web-server:compose_post_client',
            'post-storage-service:post_storage_mongo_insert_client',
            'post-storage-service:post_storage_read_posts_server',
            'post-storage-service:store_post_server',
            'social-graph-service:follow_server',
            'social-graph-service:follow_with_username_server',
            'social-graph-service:get_followees_server',
            'social-graph-service:get_followers_server',
            'social-graph-service:mongo_update_client',
            'social-graph-service:social_graph_mongo_delete_client',
            'social-graph-service:social_graph_mongo_find_client',
            'social-graph-service:social_graph_mongo_update_client',
            'social-graph-service:social_graph_redis_get_client',
            'social-graph-service:social_graph_redis_insert_client',
            'social-graph-service:social_graph_redis_update_client',
            'social-graph-service:unfollow_server',
            'social-graph-service:unfollow_with_username_server',
            'text-service:compose_text_server',
            'text-service:compose_urls_client',
            'text-service:compose_user_mentions_client',
            'unique-id-service:compose_unique_id_server',
            'url-shorten-service:compose_urls_server',
            'user-mention-service:compose_user_mentions_server',
            'user-service:compose_creator_server',
            'user-service:get_user_id_server',
            'user-service:login_server',
            'user-service:register_user_server',
            'user-service:user_mmc_get_client',
            'user-service:user_mmc_get_user_id_client',
            'user-timeline-service:write_user_timeline_mongo_insert_client',
            'user-timeline-service:write_user_timeline_redis_update_client',
            'user-timeline-service:write_user_timeline_server'
        ]
