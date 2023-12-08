import requests
import torch


feature_paths = [
    "./../storage/fixtures/datasets/simbot/1010343_Pickup_and_Deliver_Bowl_01_1_KitchenCounterTop_02_10001_AP_Prop_Desk_Green_10000_In_OfficeLayout3_mirror_action0.pt",
    "./../storage/fixtures/datasets/simbot/object_detection_data_v1__images_run1_OfficeLayout1__train__1021_color_action0.pt",
    "./../storage/fixtures/datasets/simbot/object_detection_data_v1__images_run1_OfficeLayout1__train__1022_color_action0.pt",
    "./../storage/fixtures/datasets/simbot/object_detection_data_v1__images_run1_OfficeLayout1__train__1025_color_action0.pt",
]
frames = []
for path in feature_paths:
    features = torch.load(path)
    frame_features = features["frames"][0]["features"]

    for k, v in frame_features.items():
        if isinstance(v, torch.Tensor):
            frame_features[k] = v.cpu().numpy().tolist()
    frames.append(frame_features)

radio_request = {
    "dialogue_history": [{"role": "user", "utterance": "find the radio"}],
    "environment_history": [
        {"features": [frames[0]], "output": ""},
        {"features": [frames[0], frames[1], frames[2], frames[3]], "output": ""},
        {"features": [frames[2]], "output": ""},
    ],
}
radio_response = requests.post("http://0.0.0.0:6000/grab_from_history", json=radio_request)
assert int(radio_response.content) == 0  # radio found in frame[0]

radio_request2 = {
    "dialogue_history": [{"role": "user", "utterance": "find the radio"}],
    "environment_history": [
        {"features": [frames[1]], "output": ""},
        {"features": [frames[0], frames[1], frames[2], frames[3]], "output": ""},
        {"features": [frames[2]], "output": ""},
    ],
}
radio_response2 = requests.post("http://0.0.0.0:6000/grab_from_history", json=radio_request2)
assert int(radio_response2.content) == 1  # no radio in frame[1]


laser_request = {
    "dialogue_history": [{"role": "user", "utterance": "find the laser"}],
    "environment_history": [
        {"features": [frames[0]], "output": ""},
        {"features": [frames[0], frames[1], frames[2], frames[3]], "output": ""},
        {"features": [frames[2]], "output": ""},
    ],
}
laser_response = requests.post("http://0.0.0.0:6000/grab_from_history", json=laser_request)
assert laser_response.content.decode("utf-8") == "null"
