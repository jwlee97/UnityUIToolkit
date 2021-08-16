import os
import zmq
import json
import UIoptimizer as ui
import numpy as np

def get_panel_info(request):
    info = []
    color_harmony_template = 93.6
    img_dim = [504, 896]
    panel_dim = []
    img_buff_file_name = request['imageBufferFile']
    img_meta_file_name = request['imageMetaFile']
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

    opt = ui.UIOptimizer(byte_arr, meta_data, np.array(img_dim), np.array(panel_dim), num_panels, occlusion,
                         colorfulness, edgeness, fitts_law, ce, muscle_act, rula)
        
    print("### Optimal positions for UI panels: ###")
    (labelPos, uvPlaces) = opt.weighted_optimization()
    (labelColors, textColors) = opt.color(uvPlaces)
    print("########################################\n")

    if color_harmony == True:
        colors =  opt.colorHarmony(labelColors[0], color_harmony_template)
    else:
        colors = labelColors
    
    for i in range(num_panels):
        dim_str = str(panel_dim[i][0]) + ',' + str(panel_dim[i][1])
        pos_str = str(labelPos[i][0]) + ',' + str(labelPos[i][1]) + ',' + str(labelPos[i][2])
        color_str = str(colors[i][0]) + ',' + str(colors[i][1]) + ',' + str(colors[i][2])
        text_color_str = str(textColors[i][0]) + ',' + str(textColors[i][1]) + ',' + str(textColors[i][2])
        line =  dim_str + ';' + pos_str + ';' + color_str + ';' + text_color_str
        info.append(line)

    return info

print("Connecting with Unity Toolkit...")
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

while True:
    request = socket.recv_multipart()

    if request[0].decode('utf-8') == 'P':
        req = json.loads(request[1])
        print("Received from Unity Toolkit: ", req)
        position = get_panel_info(req)
        print("Sending to Unity Toolkit: ", position)
        socket.send(json.dumps(position).encode('utf-8'))
    else:
        socket.send(b'Error')