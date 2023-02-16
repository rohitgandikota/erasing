import os 
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import glob


def make_compare_images(folder, csv_path):
    save_path = os.path.join(folder, 'compare')
    os.makedirs(save_path, exist_ok=True)

    subfolders = os.listdir(folder)

    for sub in subfolders:
        if '_xattn' in sub:
            xattn = os.path.join(folder, sub)
        elif 'selfattn' in sub:
            self = os.path.join(folder, sub)
        elif '_noxattn' in sub:
            noxattn = os.path.join(folder, sub)
        elif 'full' in sub:
            full = os.path.join(folder,sub)
        elif 'original' in sub:
            sd = os.path.join(folder,sub)
        else:
            pass

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        case_number = row.case_number
        prompt = row.prompt
        xattn_images = glob.glob(xattn+f'/{case_number}_*')
        sd_images = glob.glob(sd+f'/{case_number}_*')
        for sd_im, xattn_im in zip(sd_images, xattn_images):
            assert sd_im.split('/')[-1] == xattn_im.split('/')[-1]
            selfattn_im = xattn_im.replace('xattn','selfattn')
            noxattn_im = xattn_im.replace('xattn','noxattn')
            full_im = xattn_im.replace('xattn','full')

            fig = plt.figure(figsize = (25,7))
            plt.subplot(1,5,1)
            plt.imshow(Image.open(sd_im))
            plt.axis('off')
            plt.title('Original SD', fontsize=15)

            plt.subplot(1,5,2)
            plt.imshow(Image.open(full_im))
            plt.axis('off')
            plt.title('All Layers ESD', fontsize=15)

            plt.subplot(1,5,3)
            plt.imshow(Image.open(noxattn_im))
            plt.axis('off')
            plt.title('No XAttn ESD', fontsize=15)

            plt.subplot(1,5,4)
            plt.imshow(Image.open(selfattn_im))
            plt.axis('off')
            plt.title('Self Attn ESD', fontsize=15)

            plt.subplot(1,5,5)
            plt.imshow(Image.open(xattn_im))
            plt.axis('off')
            plt.title('Xttn ESD', fontsize=15)

            #plt.show()
            #plt.tight_layout()
            fig.suptitle(prompt, fontsize=20, y=.85)
            #plt.tight_layout()
            plt.savefig(f"{save_path}/{prompt.replace(' ','_')[:50]}_{sd_im.split('/')[-1]}", bbox_inches='tight')
            plt.close()
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'compareImages',
                    description = 'Compare different training methods and make pretty images')
    parser.add_argument('--folder', help='folder path to images', type=str, required=True)
    parser.add_argument('--csv_path', help='path to csv file with prompts', type=str, required=True)

    args = parser.parse_args()

    folder = args.folder
    csv_path = args.csv_path

    make_compare_images(folder, csv_path)
