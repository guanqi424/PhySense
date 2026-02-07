import json
import os
import plotly.graph_objects as go
from tqdm import tqdm
import pickle
import random
import copy
import pandas as pd
import argparse

global xmin, ymin, zmin, xmax, ymax, zmax
xmax = None
ymax = None
zmax = None
xmin = None
ymin = None
zmin = None

def classify_tracking_ids_with_empty_frames(frames):
    tracking_dict = {}
    result = []

    for frame_index, frame in enumerate(frames):
        if not frame:
            for key in list(tracking_dict.keys()):
                result.append(tracking_dict[key]['data'])
                del tracking_dict[key]
            continue

        for data in frame:
            tracking_id = data['tracking_id']

            if tracking_id in tracking_dict:
                if tracking_dict[tracking_id]['last_frame'] == frame_index - 1:
                    tracking_dict[tracking_id]['data'].append((frame_index, data))
                    tracking_dict[tracking_id]['last_frame'] = frame_index
                else:
                    result.append(tracking_dict[tracking_id]['data'])
                    tracking_dict[tracking_id] = {'last_frame': frame_index, 'data': [(frame_index, data)]}
            else:
                tracking_dict[tracking_id] = {'last_frame': frame_index, 'data': [(frame_index, data)]}

    for tracking_data in tracking_dict.values():
        result.append(tracking_data['data'])

    return result


def get_box_vertices(center, size):
    global xmin, ymin, zmin, xmax, ymax, zmax
    x, y, z = center
    dx, dy, dz = size
    dx, dy, dz = dx / 2, dy / 2, dz / 2
    xmax = max(xmax or x + dx, x + dx)
    ymax = max(ymax or y + dy, y + dy)
    zmax = max(zmax or z + dz, z + dz)
    xmin = min(xmin or x - dx, x - dx)
    ymin = min(ymin or y - dy, y - dy)
    zmin = min(zmin or z - dz, z - dz)
    return [
        (x - dx, y - dy, z - dz), (x + dx, y - dy, z - dz),
        (x + dx, y + dy, z - dz), (x - dx, y + dy, z - dz),
        (x - dx, y - dy, z + dz), (x + dx, y - dy, z + dz),
        (x + dx, y + dy, z + dz), (x - dx, y + dy, z + dz)
    ]


def draw_box(center, size, color, text):
    vertices = get_box_vertices(center, size)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
    ]

    x_lines, y_lines, z_lines = [], [], []

    for edge in edges:
        for vertex in edge:
            x_lines.append(vertices[vertex][0])
            y_lines.append(vertices[vertex][1])
            z_lines.append(vertices[vertex][2])
        x_lines.append(None)  # End of segment
        y_lines.append(None)
        z_lines.append(None)

    return go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=color), text=text)


def frames2Animation(sceneNum, framesBBoxs, file_path):
    fig = go.Figure()
    max_trace_num = max(len(sublist) for sublist in framesBBoxs)

    for _ in range(max_trace_num):
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=0, opacity=0)
        ))

    fig.update_layout(
        title=f"Scene {sceneNum} Animation",
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True,
                                "transition": {"duration": 0}}],
            }],
            "direction": "left",
            "showactive": False,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    frames = []
    for i, frame_data in enumerate(framesBBoxs):
        frame_traces = [draw_box(box[0][1]['translation'], box[0][1]['size'], box[1], box[0][1]['tracking_name'] + str(box[2])) for box in frame_data]
        while len(frame_traces) < max_trace_num:
            frame_traces.append(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=0, opacity=0)
            ))
        frames.append(go.Frame(data=frame_traces, name=str(i)))

    fig.frames = frames

    sliders = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 12},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [{"method": "animate", "args": [[f.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]} for f in frames]
    }

    x_length = xmax - xmin
    y_length = ymax - ymin
    z_length = zmax - zmin
    max_length = max(x_length, y_length, z_length)
    aspect_ratio = dict(x=x_length / max_length, y=y_length / max_length, z=z_length / max_length)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[xmin, xmax], autorange=False),
            yaxis=dict(range=[ymin, ymax], autorange=False),
            zaxis=dict(range=[zmin, zmax], autorange=False),
            aspectmode='manual',
            aspectratio=aspect_ratio
        ),
        sliders=[sliders])

    fig.update_layout(sliders=sliders)
    fig.write_html(file_path)


def find_frame_by_token(lst, feature_value):
    for item in lst:
        if item['token'] == feature_value:
            return item
    return None


def main(args):
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.tracking_result_path, 'r') as file:
        tracking_result = json.load(file)

    tracking_result = tracking_result['results']

    with open(args.frame_token_path_mapping_path, 'rb') as file:
        frame_token_path_mapping = pickle.load(file)

    scene_token_chain = []
    for mapping in frame_token_path_mapping:
        if mapping["prev"] == '':
            mymapping = copy.deepcopy(mapping)
            scene_token_chain.append([])
            while True:
                scene_token_chain[-1].append(mymapping["token"])
                if mymapping["next"] == '':
                    break
                mymapping = find_frame_by_token(frame_token_path_mapping, mymapping["next"])

    scene_frames = []
    for scene in scene_token_chain:
        scene_frames.append([tracking_result[mytoken] for mytoken in scene])

    scene_instance = []

    for scene in scene_frames:
        scene_instance.append(classify_tracking_ids_with_empty_frames(scene))

    for sceneNum, scene in tqdm(enumerate(scene_instance), total=len(scene_instance)):
        mySceneFrames = [[] for _ in range(len(scene_frames[sceneNum]))]

        instanceColorMapping = {}
        curFrameInstances = []

        colorStack = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
            "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
            "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0",
            "#400000", "#004000", "#000040", "#404000", "#400040", "#004040",
            "#200000", "#002000", "#000020", "#202000", "#200020", "#002020",
            "#600000", "#006000", "#000060", "#606000", "#600060", "#006060",
            "#A00000", "#00A000", "#0000A0", "#A0A000", "#A000A0", "#00A0A0",
            "#E00000", "#00E000", "#0000E0", "#E0E000", "#E000E0", "#00E0E0",
            "#FF8080", "#80FF80", "#8080FF", "#FFFF80", "#FF80FF", "#80FFFF",
            "#FFC0C0", "#C0FFC0", "#C0C0FF", "#FFFFC0", "#FFC0FF", "#C0FFFF",
            "#700000", "#007000", "#000070", "#707000", "#700070", "#007070",
            "#B00000", "#00B000", "#0000B0", "#B0B000", "#B000B0", "#00B0B0",
            "#D00000", "#00D000", "#0000D0", "#D0D000", "#D000D0", "#00D0D0",
            "#FFA0A0", "#A0FFA0", "#A0A0FF", "#FFFFA0", "#FFA0FF", "#A0FFFF",
            "#A0A0A0", "#505050", "#303030", "#101010", "#909090", "#B0B0B0",
            "#A07070", "#70A070", "#7070A0", "#A0A070", "#A070A0", "#70A0A0",
            "#80A0A0", "#A080A0", "#A0A080", "#8080A0", "#A08080", "#80A080",
            "#FF5050", "#50FF50", "#5050FF", "#FFFF50", "#FF50FF", "#50FFFF",
            "#FF3030", "#30FF30", "#3030FF", "#FFFF30", "#FF30FF", "#30FFFF",
            "#FFB0B0", "#B0FFB0", "#B0B0FF", "#FFFFB0", "#FFB0FF", "#B0FFFF",
            "#B0B0B0", "#707070", "#505050", "#303030", "#101010", "#909090",
            "#B07070", "#70B070", "#7070B0", "#B0B070", "#B070B0", "#70B0B0",
            "#80B0B0", "#B080B0", "#B0B080", "#8080B0", "#B08080", "#80B080",
            "#FF7070", "#70FF70", "#7070FF", "#FFFF70", "#FF70FF", "#70FFFF",
            "#FF4040", "#40FF40", "#4040FF", "#FFFF40", "#FF40FF", "#40FFFF",
            "#FF2020", "#20FF20", "#2020FF", "#FFFF20", "#FF20FF", "#20FFFF",
            "#FFA070", "#70FFA0", "#A070FF", "#A0FF70", "#70A0FF", "#FF70A0",
            "#A070FF", "#70A0FF", "#FF70A0", "#FFA070", "#70FFA0", "#A0FF70",
            "#B0A070", "#70B0A0", "#A070B0", "#A0B070", "#70A0B0", "#B070A0",
            "#D070A0", "#A0D070", "#70A0D0", "#D0A070", "#70D0A0", "#A070D0",
            "#F070A0", "#A0F070", "#70A0F0", "#F0A070", "#70F0A0", "#A070F0",
            "#9070A0", "#A09070", "#70A090", "#9070A0", "#70A090", "#A07090",
            "#E080A0", "#A0E080", "#80A0E0", "#E0A080", "#80E0A0", "#A080E0"
        ]

        random.shuffle(colorStack)

        for i in range(len(mySceneFrames)):
            for tracked in curFrameInstances:
                presentFlag = any(instanceFrame[0] >= i for instanceFrame in scene[tracked])
                if not presentFlag:
                    colorStack.append(instanceColorMapping[tracked])
            curFrameInstances = []

            for j in range(len(scene)):
                for instanceFrame in scene[j]:
                    if instanceFrame[0] == i:
                        curFrameInstances.append(j)
                        if j in instanceColorMapping:
                            instanceColor = instanceColorMapping[j]
                        else:
                            instanceColor = colorStack.pop(0)
                            instanceColorMapping[j] = instanceColor
                        mySceneFrames[i].append((instanceFrame, instanceColor, instanceFrame[1]['tracking_id']))
                        break

        global xmin, ymin, zmin, xmax, ymax, zmax
        xmax = None
        ymax = None
        zmax = None
        xmin = None
        ymin = None
        zmin = None

        frames2Animation(sceneNum, mySceneFrames, args.output_path+ '/scene' + str(sceneNum) + 'animation.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and visualize tracking data.")
    parser.add_argument('--tracking_result_path', type=str, default='./data/nusc/tracking_result_qd3dt.json', help='Path to the tracking result JSON file.')
    parser.add_argument('--verify_token_timestamp_path', type=str, default='./data/nusc/verify_token_timestamp.pkl', help='Path to the verify token timestamp pickle file.')
    parser.add_argument('--frame_token_path_mapping_path', type=str, default='./data/nusc/frame-token-path-mapping.pkl', help='Path to the frame token path mapping pickle file.')
    parser.add_argument('--output_path', type=str, default='./results/scenes_html', help='Path to save the output HTML animation.')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
