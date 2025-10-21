import json
import re
import os

def load_json(file_path):
    """加载JSON文件内容"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def save_json(data, file_path):
    """保存数据到JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        print(f"成功保存处理后的内容到 '{file_path}'")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def process_json_with_regex(json_data):
    
    # 处理每个元素
    for i, element in json_data.items():
        if not isinstance(element, list):
            continue
        new_strings = []
        
        # 对每个字符串应用正则表达式
        for text in element:
            if not isinstance(text, str):
                continue
            
            flag = False
            # java
            if re.match(r"^severity: INFO, message: \[Recv ListRecommendations\] product ids=\['[A-Z0-9]{10}', '[A-Z0-9]{10}', '[A-Z0-9]{10}', '[A-Z0-9]{10}', '[A-Z0-9]{10}'\]$", text):
                flag = True
                masked_text = re.sub(r"'[A-Z0-9]{10}'", "ID", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            if re.match(r"^Request finished in <:NUM:>\.[0-9]{3,4}ms <:NUM:> application\/grpc$", text):
                flag = True
                masked_text = re.sub(r"[0-9]{3,4}", "NUM", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            if re.match(r"^EVERE: Exception while executing runnable io\.grpc\.internal\.ServerImpl\$JumpToApplicationThreadServerStreamListener\$1HalfClosed\@[a-z0-9]{4,8}$", text):
                flag = True
                masked_text = re.sub(r"[a-z0-9]{4,8}", "ID", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)      
            
            # c#
            if re.match(r"^etCartAsync called with userId=([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)$", text):
                flag = True
                masked_text = re.sub(r"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)", "userId", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
               
            if re.match(r"^ddItemAsync called with userId=([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)\, productId=[A-Z0-9]{10}\, quantity=<:NUM:>$", text):
                flag = True
                masked_text = re.sub(r"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)", "userId", text)
                masked_text = re.sub(r"[A-Z0-9]{10}", "productId", masked_text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            
            if re.match(r"^rpc\.Core\.RpcException: Status\(StatusCode=\"FailedPrecondition\"\, Detail=\"Can't access cart storage\. StackExchange\.Redis\.RedisTimeoutException: Timeout awaiting response \(outbound=.*\)\, command=.*$", text):
                flag = True
                masked_text = re.sub(r"\(outbound=.*\)\, command=.*", "RESP", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            
            # Node.js
            if re.match(r"^severity: info\, message: Transaction processed: visa ending <:NUM:> Amount: (JPY|CAD|EUR|USD)[0-9]{1,7}\.<:NUM:>$", text):
                flag = True
                masked_text = re.sub(r"[0-9]{1,7}\.", "amount", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            if re.match(r"^severity: info\, message: PaymentService\#Charge invoked with request \{.*\}$", text):
                flag = True
                masked_text = re.sub(r"\{.*\}", ":chargeInfo", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            # Go
            if re.match(r"^severity: info\, message: payment went through \(transaction id: ([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)\)$", text):
                flag = True
                masked_text = re.sub(r"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)", "transactionId", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
                              
                    
            if re.match(r"^severity: info\, message: \[PlaceOrder\] user id=\"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)\" user currency=\"[A-Z]{3}\"$", text):
                flag = True
                masked_text = re.sub(r"\"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)\"", "userId", text)
                masked_text = re.sub(r"\"[A-Z]{3}\"", "cur", masked_text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
                               
                    
            if re.match(r"^(mptyCartAsync|etCartAsync) called with userId=([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)$", text):
                flag = True
                masked_text = re.sub(r"([a-z0-9]{8}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{4}|<:NUM:>)\-([a-z0-9]{12}|<:NUM:>)", "userId", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
                    
            if re.match(r"^Request finished in <:NUM:>\.[0-9]{1,4}ms <:NUM:> application\/grpc$", text):
                flag = True
                masked_text = re.sub(r"[0-9]{1,4}", "time", text)
                if masked_text not in new_strings:
                    new_strings.append(masked_text)
            
            if not flag:
                new_strings.append(text)
        
        # 添加新字符串到列表
        json_data[i] = new_strings
    
    return json_data

def main():
    """主函数"""
    # JSON文件路径
    file_path = "/root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/analysis/log/log_patterns.json"
    
    # 加载JSON数据
    json_data = load_json(file_path)
    if json_data is None:
        return
    
    # 处理数据
    processed_data = process_json_with_regex(json_data)
    
    # 保存处理后的数据
    save_json(processed_data, file_path)

if __name__ == "__main__":
    main()



# import re

# # 示例日志行
# log_lines = [
#     "severity: info, message: [PlaceOrder] user id=\"b87d4b03-4bb2-40b8-be74-9c3a4bbd60eb\" user currency=\"JPY\"",
#     "severity: info, message: [PlaceOrder] user id=\"6a63915a-<:NUM:>-<:NUM:>-b2df-9d13304428bb\" user currency=\"EUR\"",
#     "severity: info, message: [PlaceOrder] user id=\"7233e639-de04-<:NUM:>-a7d7-4cf1c5a2b0d3\" user currency=\"JPY\"",
#     "severity: info, message: [PlaceOrder] user id=\"c261b386-<:NUM:>-4e0d-b108-fa5353cad029\" user currency=\"CAD\"",
#     "severity: info, message: [PlaceOrder] user id=\"1bd4a3f9-eae2-<:NUM:>-<:NUM:>-1189aee0f20f\" user currency=\"USD\""
# ]

# # 定义正则表达式模式 - 匹配完整的日志行结构，使用括号捕获产品ID部分
# pattern = r"^severity: info\, message: \[PlaceOrder\] user id=\"[a-z0-9]{8}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{12}\" user currency=\"[A-Z]{3}\"$"

# # 替换函数
# def replace_ids(log_line):
#     # 检查是否匹配完整的日志行格式
#     if re.match(pattern, log_line):
#         # 只替换产品ID部分为ID
#         return re.sub(r"\"[a-z0-9]{8}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{12}\"", "ID", log_line)
#     # 如果不匹配完整格式，返回原日志行
#     return log_line

# # 处理每一行日志
# for log_line in log_lines:
#     print(replace_ids(log_line))

