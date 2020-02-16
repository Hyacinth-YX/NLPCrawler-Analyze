dict = {
	"s": {"fg": "#A020F0", "note": "客体"},
	"z": {"fg": "black", "note": "主体"},
	"t": {"fg": "orange", "note": "关系词"},
}

import json

with open("config.json", 'w') as f:

	f.write(json.dumps(dict))