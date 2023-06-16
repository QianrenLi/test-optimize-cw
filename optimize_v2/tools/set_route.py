import psutil
from ipaddress import ip_network
import argparse
import os

## Setup route table for Intel card and Realtek card
def two_IC_setup():
    info = init_info()
    if_name_Intel = "wlp"
    if_name_Realtek = "wlx"
    rets = []
    rets.append(route_setup(info, if_name_Intel, 111))
    rets.append(route_setup(info, if_name_Realtek, 112))
    print("Setup Failed") if -1 in rets else print("Setup Successful")
    


## init ip information
def init_info():
    addrs = psutil.net_if_addrs()
    ## from addrs to info name
    info = {}
    for key in addrs.keys():
        for addr in addrs[key]:
            if addr.family == 2:
                info[key] = addr.address
    return info

## Setup route table
def route_setup(info, if_type, table_id, netmask=24):
    for if_name in info.keys():
        if if_type in if_name:
            ip_addr = ip_network(info[if_name] + "/" + str(netmask), strict=False)
            # get default first host as gateway from ip_addr
            gateway = str(ip_addr[1])
            cmd = "sudo ip route add " + str(ip_addr) + " dev " + if_name + " proto kernel scope link src " + info[if_name] + " table " + str(table_id)
            cmd = cmd + " && sudo ip route add default via " + gateway + " table " + str(table_id)
            cmd = cmd + " && sudo ip rule add from " + info[if_name] + " table " + str(table_id)
            # Execute cmd in bash
            run_cmd(cmd)
            return 1
    return -1

## Subprocess run cmd
def run_cmd(cmd):
    os.system(cmd)

def sys_conf_init():
    path = "/etc/sysctl.conf"
    info = init_info()
    if not if_conf_exist(path):
        write_sys_conf(info, path)
    else:
        print("Config Exists")

def sys_conf_exit():
    path = "/etc/sysctl.conf"
    if if_conf_exist(path):
        remove_last_k_lines(path, 10)
    else:
        print("Config not exists")

def if_conf_exist(path):
    with open(path, "r") as f:
        for line in f.readlines():
            if "net.ipv4.conf.all.all_announce = 2" in line:
                return True
    return False

## Write to /etc/sysctl.conf
def write_sys_conf(info, path):
    if_name_Intel = None
    if_name_Realtek = None
    for if_name in info.keys():
        if_name_Intel = if_name if "wlp" in if_name else if_name_Intel
        if_name_Realtek = if_name if "wlx" in if_name else if_name_Realtek
    print([if_name_Intel, if_name_Realtek])
    if None not in [if_name_Intel, if_name_Realtek]:
        print("start write")
        with open(path, "a") as f:
            f.write("\nnet.ipv4.conf.all.all_announce = 2\n")               #in remove last k lines function, it will additionally remove "\n"
            f.write("net.ipv4.conf.all.arp_ignore = 1\n")
            f.write("net.ipv4.conf.default.accept_source_route = 0\n")
            f.write("net.ipv4.conf.default.arp_announce = 2\n")
            f.write("net.ipv4.conf.default.arp_ignore = 1\n")
            f.write("net.ipv4.conf.default.rp_filter = 1\n")
            ## Write Intel card
            f.write("net.ipv4.conf.%s.arp_announce = 2\n" % if_name_Intel)
            f.write("net.ipv4.conf.%s.arp_ignore = 1\n" % if_name_Intel)
            ## Write Realtek card
            f.write("net.ipv4.conf.%s.arp_announce = 2\n" % if_name_Realtek)
            f.write("net.ipv4.conf.%s.arp_ignore = 1\n" % if_name_Realtek)
    else:
        print("write failed")

def remove_last_k_lines(path, k):
    with open(path, "r+") as f:
        f.seek(0, 2)

        pos = f.tell()
        for count in range(k):
            pos -= 1
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, 0)

        if pos > 0:
            f.seek(pos, 0)
            f.truncate()
            print("Remove last " + str(k) + " lines")
        else:
            print("Remove fails -- not enough lines")

def main(args):
    if args.start:
        two_IC_setup()
        sys_conf_init()
    if args.exit:
        sys_conf_exit()


## Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", action="store_true", help = "Start config and cmd setup"
    )
    parser.add_argument(
        "--exit", action = "store_true", help  = "Exit config"
    )
    args = parser.parse_args()
    main(args)
    # test_cmd = "ls"
    # subprocess.run(test_cmd, shell=True)
    # remove_last_k_lines("test.txt", 3)
    # sys_conf_init()
    # sys_conf_exit()
