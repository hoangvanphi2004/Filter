import xml.etree.ElementTree as ET
from torchvision.transforms import ToTensor;
from PIL import Image;
import pandas;

df = pandas.DataFrame([], columns = ['box-left', 'box-top', 'box-width', 'box-height', 'img-width', 'img-height', 'img-path']);

data = ET.parse('cs194-26-fa20-proj4\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_train.xml');
data = data.getroot().find('images').findall('image');

for el in data:
    image = Image.open('cs194-26-fa20-proj4/ibug_300W_large_face_landmark_dataset/' + el.attrib['file']);
    image = ToTensor()(image);
    box = el.find('box');
    label = {};
    label['box'] = [int(box.attrib['left']), int(box.attrib['top']), int(box.attrib['width']), int(box.attrib['height'])];
    row = label['box'] + [image.size()[1], image.size()[2], 'cs194-26-fa20-proj4/ibug_300W_large_face_landmark_dataset/' + el.attrib['file']];
    df = pandas.concat([df, pandas.DataFrame([row], columns = ['box-left', 'box-top', 'box-width', 'box-height', 'img-width', 'img-height', 'img-path'])], ignore_index=True);
print(df.head())
df.to_csv("./ImageSizeDataFrame.csv", index=False, header=True);