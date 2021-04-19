import xml.etree.ElementTree as ET
import json


def read_danmaku_file(file_path, guard=False):
    tree = ET.parse(file_path)
    root = tree.getroot()

    all_children = [child for child in root if child.tag in ['gift', 'sc', 'd'] + (['guard'] if guard else [])]
    return all_children


def get_time(child: ET.Element):
    # noinspection PyBroadException
    try:
        if child.tag == 'd':
            return float(child.attrib['p'].split(',')[0])
        elif child.tag == 'gift' or child.tag == 'sc':
            return float(child.attrib['ts'])
    except:
        print(f"error getting time from {child}")
        return 0


def get_value(child: ET.Element):
    # noinspection PyBroadException
    try:
        if child.tag == 'd':
            return 1
        elif child.tag == 'gift':
            raw_data = json.loads(child.attrib['raw'])
            return raw_data['total_coin'] / 1000 / 10
        elif child.tag == 'sc':
            raw_data = json.loads(child.attrib['raw'])
            return raw_data['price'] / 10
        elif child.tag == 'guard':
            raw_data = json.loads(child.attrib['raw'])
            return raw_data['price'] / 1000 / 10
    except:
        print(f"error getting value from {child}")
        return 0
