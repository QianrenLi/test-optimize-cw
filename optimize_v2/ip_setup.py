from tap import Connector
import json

names = Connector().list_all()
clients = [ Connector(n) for n in names ]

#extract p2p ip
conn = Connector()
ip_table = {}
for c in clients:
    print(c.client)
    ip_table.update({c.client: {}})
    conn.batch(c.client, 'read_ip_addr', {"keyword": "p2p\|wlan"})

outputs = conn.executor.wait(1).fetch().apply()
results = [ o['ip_addr'] for o in outputs ]

for r,c in zip(results,clients):
    try:
        ipv4_addrs = eval(r)
    except:
        print("Error: client %s do not exist valid ipv4 addr" % c.client)
    else:
        # print(ipv4_addrs)
        # Generally multiple ipv4 with same name might be detect, but we only need the first one
        for ipv4_addr in ipv4_addrs[::-1]:
            ip_table[c.client].update({ipv4_addr[0]: ipv4_addr[1]})

# print(ip_table)

# save the dict into json file
with open('./temp/ip_table.json', 'w') as f:
    json.dump(ip_table, f)
