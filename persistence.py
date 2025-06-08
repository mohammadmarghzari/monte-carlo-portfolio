import json

def save_portfolio_json(config_dict):
    return json.dumps(config_dict, ensure_ascii=False)

def load_portfolio_json(json_str):
    return json.loads(json_str)
