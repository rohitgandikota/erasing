from nudenet import NudeClassifier, NudeDetector
import imageio
from PIL import Image
import pandas as pd
import numpy as np
import os
import glob
import tqdm
import argparse

def detectNudeClasses(image_paths, csv_path, save_path):
    detector = NudeDetector()
    df = pd.read_csv(csv_path)
    df['NudeNet_label'] = np.zeros(len(df),dtype=float)
    for image in image_paths:
        print(image)
        case_number = int(image.split('/')[-1].split('_')[0].replace('.png',''))
        detected = detector.detect(image)
        label = ''
        for detect in detected:
            label = label + detect['label']+'-'
        df.loc[df.case_number==case_number, f'NudeNet_label'] = label[:-1]
        print(df.loc[df.case_number==case_number, f'NudeNet_label'])
    df.to_csv(save_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'NudeNet Classes',
                    description = 'Get Nudenet classes for images')
    parser.add_argument('--folder', help='folder path to images', type=str, required=True)
    parser.add_argument('--csv_path', help='path to csv file with prompts', type=str, required=False, default='/share/u/rohit/stable-diffusion/unsafe-prompts4703.csv')
    parser.add_argument('--save_path', help='path to csv file with prompts', type=str, required=False, default=None)

    args = parser.parse_args()
    
    csv_path = args.csv_path 
    folder = args.folder
    save_path = args.save_path
    if save_path is None:
        name = folder.split('/')[-1]
        save_path = f'{folder}/{name}_NudeClasses.csv'
    
    image_paths = glob.glob(f'{folder}/*.png')
    
    detectNudeClasses(image_paths, csv_path, save_path)
