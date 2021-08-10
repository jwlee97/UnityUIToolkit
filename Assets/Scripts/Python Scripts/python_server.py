import zmq
import json
import UIoptimizer as ui
import numpy as np

def get_panel_info(request):
    info = []
    color_harmony_template = 93.6
    img_dim = [504, 896]
    panel_dim = []
    image_file = request['imageBufferFile']
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

    f = open(image_file, "r")
    byte_arr = bytes(f.read(), 'utf-8')

    opt = ui.UIOptimizer(byte_arr, np.array(img_dim), np.array(panel_dim), num_panels, occlusion,
                            colorfulness, edgeness, fitts_law, ce, muscle_act, rula)
        
    (labelPos, uvPlaces) = opt.weighted_optimization()
    (labelColors, textColors) = opt.color(uvPlaces)

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

context = zmq.Context()
print("Connecting with Unity toolkit...")
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

while True:
    request = socket.recv_multipart()

    if request[0].decode('utf-8') == 'C':
        req = json.loads(request[1])
        print("Received: ", req)
        position = get_panel_info(req)
        print("Sending: ", position)
        socket.send(json.dumps(position).encode('utf-8'))
    else:
        socket.send(b'Error')