import os
import zmq
import json
import UIoptimizer as ui
import numpy as np
from PreferenceLearning import test as pltest

def get_request_info(request):
    panel_dim = []
    img_dim = [504, 896]
    img_buff_file_name = request["imageBufferFile"]
    img_meta_file_name = request["imageMetaFile"]
    num_panels = request["numPanels"]
    occlusion = request["occlusion"]
    color_harmony = request["colorHarmony"]
    constraints = request["constraints"]
    colorfulness = request["colorfulness"]
    edgeness = request["edgeness"]
    fitts_law = request["fittsLaw"]
    ce = request["ce"]
    muscle_act = request["muscleActivation"]
    rula = request["rula"]

    for c in constraints:
        panel_dim.append((c["height"], c["width"]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_buff_file = dir_path + "\\input_images\\" + img_buff_file_name
    img_meta_file = dir_path + "\\input_images\\" + img_meta_file_name

    f = open(img_buff_file, "r")
    byte_arr = bytes(f.read(), 'utf-8')

    with open(img_meta_file, 'r') as f:
        meta_data = f.read()

    optimizer = ui.UIOptimizer(byte_arr, meta_data, np.array(img_dim), np.array(panel_dim), num_panels, occlusion,
                               color_harmony, colorfulness, edgeness, fitts_law, ce, muscle_act, rula)

    return optimizer, panel_dim, byte_arr


def get_panel_loc(optimizer, panel_dim):
    info = []
    color_harmony_template = 93.6

    print("\n### Optimal positions for UI panels: ###")
    (labelPos, uvPlaces) = optimizer.weighted_optimization()
    (labelColors, textColors) = optimizer.color(uvPlaces)
    print("########################################\n")

    if optimizer.colorHarmony == True:
        colors =  optimizer.getColorHarmony(labelColors[0], color_harmony_template)
    else:
        colors = labelColors
    
    for i in range(optimizer.num_panels):
        dim_str = str(panel_dim[i][0]) + ',' + str(panel_dim[i][1])
        pos_str = str(labelPos[i][0]) + ',' + str(labelPos[i][1]) + ',' + str(labelPos[i][2])
        color_str = str(colors[i][0]) + ',' + str(colors[i][1]) + ',' + str(colors[i][2])
        text_color_str = str(textColors[i][0]) + ',' + str(textColors[i][1]) + ',' + str(textColors[i][2])
        lower_bounds = str(optimizer.xl[0]) + ',' + str(optimizer.xl[1]) + ',' + str(optimizer.xl[2])
        upper_bounds = str(optimizer.xu[0]) + ',' + str(optimizer.xu[1]) + ',' + str(optimizer.xu[2])
        line =  dim_str + ';' + pos_str + ';' + color_str + ';' + text_color_str + ';' + lower_bounds + ';' + upper_bounds
        info.append(line)

    return info

def launch_pref_learning(optimizer, byte_arr):
    info = []
    labelPos, colors = pltest.toolkit(byte_arr, optimizer)

    for i in range(optimizer.num_panels):
        pos_str = str(labelPos[i][0]) + ',' + str(labelPos[i][1]) + ',' + str(labelPos[i][2])
        color_str = str(colors[i][0]) + ',' + str(colors[i][1]) + ',' + str(colors[i][2])
        line =  pos_str + ';' + color_str + '\n'
        info.append(line)

    return info

print("Connecting with Unity Toolkit...")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

while True:
    request = socket.recv_multipart()
    req = json.loads(request[1])
    print("Received from Unity Toolkit: ", req)
    optimizer, panel_dim, byte_arr = get_request_info(req)

    if request[0].decode('utf-8') == 'P':
        info = get_panel_loc(optimizer, panel_dim)
        print("Sending to Unity Toolkit: ", info)
        socket.send(json.dumps(info).encode('utf-8'))
    elif request[0].decode('utf-8') == 'L':
        info = launch_pref_learning(optimizer, byte_arr)
        socket.send(json.dumps(info).encode('utf-8'))
    else:
        socket.send(b'Error')