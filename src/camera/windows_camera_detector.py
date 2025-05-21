import subprocess
import json
import re
from typing import List, Dict, Any, Optional

def get_cameras_from_device_manager() -> List[Dict[str, Any]]:
    cameras = []

    try:
        powershell_cmd = """
        Get-PnpDevice -Class Camera -Status OK |
        Select-Object FriendlyName, InstanceId, Status, DeviceID |
        ConvertTo-Json
        """

        process = subprocess.Popen(
            ["powershell", "-Command", powershell_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        stdout, stderr = process.communicate()

        if stderr:
            return cameras

        if stdout.strip():
            try:
                devices = json.loads(stdout)
                if isinstance(devices, dict):
                    devices = [devices]

                for i, device in enumerate(devices):
                    if not all(key in device for key in ['FriendlyName', 'InstanceId', 'Status']):
                        continue

                    if device['Status'] != 'OK':
                        continue

                    if ('Intel RealSense' in device['FriendlyName'] and
                        'Depth' in device['FriendlyName'] and
                        'RGB' not in device['FriendlyName']):
                        continue

                    camera_info = {
                        'id': i,
                        'name': device['FriendlyName'],
                        'instance_id': device['InstanceId'],
                        'device_id': device.get('DeviceID', ''),
                        'status': device['Status'],
                        'width': 1920,
                        'height': 1080,
                        'fps': 30
                    }

                    cameras.append(camera_info)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    return cameras

def map_device_to_index(instance_id: str) -> Optional[int]:
    try:
        match = re.search(r'USB\\VID_[0-9A-F]+&PID_[0-9A-F]+\\(\d+)', instance_id)
        if match:
            return int(match.group(1))

        match = re.search(r'\\(\d+)$', instance_id)
        if match:
            return int(match.group(1))

        return None
    except Exception:
        return None

def get_camera_index_map() -> Dict[str, int]:
    index_map = {}

    try:
        cameras = get_cameras_from_device_manager()

        for i, camera in enumerate(cameras):
            instance_id = camera['instance_id']
            index = map_device_to_index(instance_id)
            if index is None:
                index = i

            index_map[instance_id] = index
    except Exception:
        pass

    return index_map
