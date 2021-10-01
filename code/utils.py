import json

def read_json(mdir):
    with open(mdir, 'r') as f:
        tmp = json.loads(f.read())
    return tmp
 