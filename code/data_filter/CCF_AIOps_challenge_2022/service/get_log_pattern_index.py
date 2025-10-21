import json

def find_keyword_in_dict(d, keyword, dictkey):
    """
    查找关键字在字典中的哪个 key 对应的 list 中，并返回其下标。
    
    :param d: 输入的字典，每个 key 对应一个 list
    :param keyword: 要查找的关键字
    :return: (key, index) 如果找到；None 如果未找到
    """
    for key, value_list in d.items():
        if key == dictkey:
            if keyword in value_list:
                index = value_list.index(keyword)
                return key, index
    return None

# 读取 JSON 文件
file_path = '/root/shared-nvme/work/code/Repdf/temp_data/2022_CCF_AIOps_challenge/analysis/log/log_patterns.json'  # 替换为你的实际文件路径
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    keyword = "ocket.gaierror: [Errno -2] Name or service not known"  # 替换为你想查找的关键字
    result = find_keyword_in_dict(data, keyword, "Python")

    if result:
        print(f"关键字 '{keyword}' 在 key='{result[0]}' 的列表中，下标为 {result[1]}")
    else:
        print(f"关键字 '{keyword}' 未在字典中找到。")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查路径是否正确。")
except json.JSONDecodeError:
    print(f"文件 {file_path} 不是有效的 JSON 格式。")