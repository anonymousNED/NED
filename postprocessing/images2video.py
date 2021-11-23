import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
from moviepy.editor import *

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str, nargs='+', default='/Data/fpar/head2head/datasets/PacinoAngry/dataset/test/images/',
                        help="path to saved images")
    parser.add_argument('--out_path', type=str, default='/gpu-data/fpar/video.mp4',
                        help="path to save video")
    parser.add_argument('--fps', type=float, default=30,
                        help=".")
    parser.add_argument('--audio', type=str, default=None,
                        help="Path to original .mp4 file that contains audio")
    parser.add_argument('--layout', type=str, default='2D', choices = ['1D', '2D'])

    args = parser.parse_args()

    if len(args.imgs_path)<2:
        for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
            if len(fnames)==0:
                continue
            for name in sorted(fnames):
                im = cv2.imread(os.path.join(root, name))
                w,h = im.shape[1], im.shape[0]
                break
            break

        video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        print('Converting images to video ...')

        for root, _, fnames in sorted(os.walk(args.imgs_path[0])):
            for name in tqdm(sorted(fnames)):
                im = cv2.imread(os.path.join(root, name))
                video.write(im)

        cv2.destroyAllWindows()
        video.release()

        print('DONE')

    else:
        if args.layout == '2D':
            images = sorted(os.listdir(args.imgs_path[0]))
            im = cv2.imread(os.path.join(args.imgs_path[0], images[0]))
            w,h = im.shape[1], im.shape[0]

            nw = int(len(args.imgs_path)/2)
            if len(args.imgs_path)%2==1:
                video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(nw*w), int(3*h)))
            else:
                video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(nw*w), int(2*h)))
            print('Converting images to video ...')

            for img in tqdm(images):
                row1 = np.concatenate([cv2.imread(os.path.join(args.imgs_path[i], img)) for i in range(nw)], 1)
                row2 = np.concatenate([cv2.imread(os.path.join(args.imgs_path[i], img)) for i in range(nw,2*nw)], 1)
                if len(args.imgs_path)%2==1:
                    row3 = np.concatenate([np.full_like(im, 255) for _ in range(int(nw/2))]+[cv2.imread(os.path.join(args.imgs_path[-1], img))]+[np.full_like(im, 255) for _ in range(int(nw/2))], 1)
                    video.write(np.concatenate((row1,row2, row3), 0))
                else:
                    video.write(np.concatenate((row1,row2), 0))

            cv2.destroyAllWindows()
            video.release()
        else:
            images = sorted(os.listdir(args.imgs_path[0]))
            im = cv2.imread(os.path.join(args.imgs_path[0], images[0]))
            w,h = im.shape[1], im.shape[0]

            nw = len(args.imgs_path)
            video = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (int(nw*w), h))
            print('Converting images to video ...')

            for img in tqdm(images):
                row = np.concatenate([cv2.imread(os.path.join(args.imgs_path[i], img)) for i in range(nw)], 1)
                video.write(row)

            cv2.destroyAllWindows()
            video.release()

    if args.audio is not None:
        print('Adding audio with MoviePy ...')
        video = VideoFileClip(args.out_path)
        video_audio = VideoFileClip(args.audio)
        video = video.set_audio(video_audio.audio)
        os.remove(args.out_path)
        video.write_videofile(args.out_path)

    print('DONE')

if __name__ == "__main__":
    main()
