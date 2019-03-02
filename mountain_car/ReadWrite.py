import json


def write_q_func(q_func, f_name):
    with open(f_name, 'w') as file:
        file.write(json.dumps(q_func))


def read_q_func(f_name):
    with open(f_name) as f:
        data = json.load(f)
        for key in data:
            data[key] = {int(k): float(v) for k, v in data[key].items()}
    return data
