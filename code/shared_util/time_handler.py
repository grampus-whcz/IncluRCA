from datetime import datetime, timezone
import pytz


class TimeHandler:
    @staticmethod
    def datetime_to_timestamp(datetime_str: str) -> int:
        timezone = pytz.timezone("Asia/Shanghai")
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return int(timezone.localize(dt).timestamp())

    @staticmethod
    def timestamp_to_datetime(timestamp: int):
        timezone = pytz.timezone("Asia/Shanghai")
        dt = pytz.datetime.datetime.fromtimestamp(timestamp, timezone)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def datetime_to_timestamp_new(datetime_str: str) -> int:
        # 直接解析为 UTC 时间
        dt = datetime.strptime(datetime_str.strip('Z'), '%Y-%m-%dT%H:%M:%S.%f')
        
        # 转为时间戳（UTC 时间戳），然后 + 8小时（28800 秒）
        return int(dt.timestamp() + 8 * 3600)
    
    @staticmethod
    def datetime_to_timestamp_new1(datetime_str: str) -> int:    
        # 直接解析为 UTC 时间
        dt = datetime.strptime(datetime_str.strip('Z'), '%Y-%m-%dT%H:%M:%S')
        
        # 转为时间戳（UTC 时间戳），然后 + 8小时（28800 秒）
        return int(dt.timestamp() + 8 * 3600)
