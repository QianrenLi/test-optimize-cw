#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import time
from tap import Connector
from tap import NoResponseException

DURATION = 10
REAL_LEN_WEIGHT = 0.01
PROJ_LEN_WEIGHT = 5
THRU_WEIGHT = 0.5

Path('logs').mkdir(parents=True, exist_ok=True)
f = open( f'logs/log_{time.ctime()}.json', 'a+' )

PARAMS = {
    'target_addr':'192.168.3.34',
    'duration': DURATION
}

CONFIG = {
    'file_only':   { 'file_port':5201 },
    'real+file_A': { 'file_port':5202, 'real_port':5205 },
    'real+file_B': { 'file_port':5203, 'real_port':5206 },
    'proj+file':   { 'file_port':5204, 'proj_port':5207 }
}

CONTROL = {
    'file_only':   {'file_throttle':0},
    'real+file_A': {'file_throttle':0},
    'real+file_B': {'file_throttle':0},
    'proj+file':   {'file_throttle':0}
}

QOS = {
    'file_only':   { 'file_thru':0, 'real_len':0, 'proj_len':0 },
    'real+file_A': { 'file_thru':0, 'real_len':0, 'proj_len':0 },
    'real+file_B': { 'file_thru':0, 'real_len':0, 'proj_len':0 },
    'proj+file':   { 'file_thru':0, 'real_len':0, 'proj_len':0 }
}

def calc_qos(results):
    ## update global `QOS`
    for k,v in results.items():
        QOS[k].update(v)
    f.write( json.dumps(QOS)+',' ); f.flush()
    #print( json.dumps(QOS,indent=2) )
    ## calculate QoS
    qos_proj = PROJ_LEN_WEIGHT * sum([ float(x['proj_len']) for x in QOS.values() ])
    qos_real = REAL_LEN_WEIGHT * sum([ float(x['real_len']) for x in QOS.values() ])
    qos_file = -THRU_WEIGHT * sum([ float(x['file_thru']) for x in QOS.values() ])
    return qos_proj + qos_real + qos_file

def init():
    conn = Connector()
    clients = conn.list_all()
    ## init port
    for name in clients:
        conn.batch(name, 'port', CONFIG[name])
    ## init throttle
    for name in clients:
        conn.batch(name, 'throttle', CONTROL[name])
    conn.batch_wait(1.0).apply()

def update_control(throttle:list):
    for name,val in zip(CONTROL.keys(), throttle):
        CONTROL[name]['file_throttle'] = val
    ## apply throttle control
    conn = Connector()
    clients = conn.list_all()
    for name in clients:
        conn.batch(name, 'throttle', CONTROL[name])
    conn.batch_wait(1.0).apply()

def run_with_throttle(throttle, test_speed=False):
    update_control(throttle)
    ##
    conn = Connector()
    clients = conn.list_all()
    num_client = len(clients)
    all_params = [ {**PARAMS,**CONFIG[n]} for n in clients ]

    ##
    for name,params in zip(clients, all_params):
        _task = name.split('@')[0]
        _task = 'file_only' if test_speed else name
        conn.batch('', f'send_{_task}', params, timeout=DURATION+5)
    conn.batch_wait(2.0)
    ##
    for name,params in zip(clients, all_params):
        _task = name.split('@')[0]
        _task = 'file_only' if test_speed else name
        conn.batch(name, f'recv_{_task}', params, timeout=DURATION+3)
    conn.batch_wait(DURATION+1)
    
    while True:
        try:
            outputs = conn.batch_fetch().apply()
        except NoResponseException:
            time.sleep(1.0)
        else:
            
            break
    ##
    results = dict()
    for i,o in enumerate(outputs[:num_client]):
        results[ clients[i] ].update( o )
    for i,o in enumerate(outputs[num_client:]):
        results[ clients[i] ].update( o )
    ##
    res = calc_qos(results) if not test_speed else results
    return res
    
def run_speed_test() -> dict:
    results = run_with_throttle([0,0,0,0], test_speed=True)
    return results

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Order: ['file_only', 'real+file_A', 'real+file_B', 'proj+file']
epsilon_M = 0.5
e_I = 1
h_K = 1
alpha = 5
lower_bound = np.array([0,0,0,0])
upper_bound = np.array([250,250,250,250])

const_offset = 10.0
real_size = 60.0
proj_size = 10.0

def initial_point():
    results = run_speed_test()
    for k in results.keys():
        if k=='file_only': results[k] -= const_offset
        if k=='real+file': results[k] -= real_size
        if k=='proj+file': results[k] -= proj_size
    print(results)
    return results

def gradient_search():
    counter = 0
    x_his = np.array([0.01, 0.01,0.01,0.01])
    x = np.array([0.5, 0.5, 0.5, 0.5])
    ##
    while True:
        counter += 1
        direction = counter % len(x)
        h_fd = max(1, abs(x[direction]))*epsilon_M
        temp_x = x + h_fd * e_I
        
        ####
        print(x, temp_x)
        qos = run_with_throttle( x.tolist() )
        qos_h = run_with_throttle( temp_x.tolist() )
        print(qos, qos_h)
        ####

        delta_x = alpha * h_K * (qos_h - qos) / h_fd
        x_ = x - delta_x
        ##
        if True in (x_ <= lower_bound) or True in (x_ >= upper_bound) :
            h_delta_x = (x - x_his)*0.5;
            h_direction = np.sign(-delta_x);
            x_ = x + h_direction*abs(h_delta_x);
            x_his = x
        x = x_
        pass
    pass


if __name__=='__main__':
    # gradient_search()
    run_speed_test()
