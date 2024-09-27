import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from tqdm import tqdm

from util import *
from model import *
from data import *


def saliency_heatmap_plot(data_img, data_param, data_pos, extra_title):
    print('> temp_plot start for', extra_title)
    # Create a figure with 3 subplots (3 rows, 1 column)
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Adjust size as needed

    ''' 1. Mean saliency map over all frames '''
    print('> temp_plot for img')
    mean_saliency = np.mean(data_img, axis=0)
    cax = axs[0].imshow(mean_saliency, cmap='viridis', aspect='auto')
    axs[0].set_title(f'Mean Saliency Map - Img Feature - {extra_title}')
    axs[0].set_xlabel('Feature')
    axs[0].set_ylabel('Temporal')
    axs[0].axis('on')

    x_ticks = np.int32(np.linspace(0, data_img.shape[2], 10))
    axs[0].set_xticks(x_ticks)

    y_ticks = np.int32(np.arange(0, data_img.shape[1], 50))
    y_ticklables = y_ticks - 100
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_ticklables)

    # colorbar
    cbar = fig.colorbar(cax, ax=axs[0])

    ''' 2. Mean saliency map for param & pos map '''
    print('> temp_plot for param and pos')
    mean_param_saliency = np.mean(data_param, axis=0)  # Assuming these indices correspond to param
    mean_pos_saliency = np.mean(data_pos, axis=0)  # Adjust index range for position if needed
    mean_param_pos_saliency = np.concatenate((mean_param_saliency, mean_pos_saliency), axis=1)

    cax = axs[1].imshow(mean_param_pos_saliency, cmap='viridis', aspect='auto')
    axs[1].set_title(f'Mean Saliency Map - Param & Pos - {extra_title}')
    axs[1].set_xlabel('Feature')
    axs[1].set_ylabel('Temporal')
    axs[1].axis('on')

    # Set x-ticks
    x_ticks = np.arange(0, mean_param_pos_saliency.shape[1])
    x_ticklables = param_str_list[6:11] + pos_str_list
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_ticklables, rotation=90, ha='center')

    y_ticks = np.int32(np.arange(0, data_img.shape[1], 50))
    y_ticklables = y_ticks - 100
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_ticklables)

    # colorbar
    cbar = fig.colorbar(cax, ax=axs[1])

    ''' saving '''
    # Adjust layout
    plt.tight_layout()
    output_path = f'{machine_output_dir}/temp/saliency_plot.{extra_title}.png'
    plt.savefig(output_path, dpi=600)
    # plt.show()

def temp_video(data, extra_title):
    # Prepare to write the video
    output_video = f'{machine_output_dir}/temp/saliency_video_{extra_title}.mp4'
    print(f"\n> making video: {output_video}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    fps = 5  # Frames per second
    height, width = 800, 1600  # Adjust height and width to desired output size
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Calculate the max value for color bar limits
    max_value = np.max(data)

    # Process each frame in the tensor
    for i in tqdm(range(0, data.shape[0], 100)):    # every Nth frame
        frame_tensor = data[i]  # Get the i-th frame

        # Create a heatmap image
        plt.figure(figsize=(10, 6))  # Increased figure size for better visibility
        plt.imshow(frame_tensor, cmap='viridis', aspect='auto', vmin=0, vmax=max_value)
        plt.colorbar()
        plt.title(f'{extra_title} / {i}')
        plt.xlabel('Feature')
        plt.ylabel('Temporal')
        plt.axis('on')  # Show axes

        # Set x-ticks from the dictionary
        x_ticks = [j for j in range(data.shape[2])]

        if extra_title == 'param':
            x_tick_labels = param_str_list[6:11]
            plt.xticks(x_ticks, x_tick_labels)
        elif extra_title == 'pos':
            x_tick_labels = pos_str_list[:]
            plt.xticks(x_ticks, x_tick_labels, rotation=45, ha='right')
        else:
            plt.xticks([])

        # Save the plot to a temporary image buffer
        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight', pad_inches=0.5, dpi=600)  # Save to buffer
        plt.close()  # Close the figure
        buf.seek(0)  # Move to the beginning of the BytesIO buffer

        # Read the image from the buffer
        frame_image = plt.imread(buf)  # Read image from buffer
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        frame_image = cv2.resize(frame_image, (width, height))  # Resize to match video dimensions
        video_writer.write(frame_image)  # Write the frame to the video

    # Release the video writer
    video_writer.release()
    print(f"> video saved as {output_video}")


if __name__ == '__main__':

    # Setup local environment
    data_root_dir = r'/home/ubuntu/Desktop/mydata'
    machine_output_dirs = [os.path.join(data_root_dir, 'output/p2_ded_bead_profile/v12.0.d/240926-234148.5342.param_5_saliency.enc_200'),
                           os.path.join(data_root_dir, 'output/p2_ded_bead_profile/v12.0.d/240926-234322.8042.param_5_saliency.enc_300'),
                           os.path.join(data_root_dir, 'output/p2_ded_bead_profile/v12.0.d/240926-234346.0610.param_5_saliency.enc_400')
                           ]
    for machine_output_dir in machine_output_dirs:
        print('\n> loading data in output dir', machine_output_dir)
        # Load data and create videos
        # dataset_name = 'Low_noise_noise_1_dataset'
        dataset_exclude_for_deploy = ['Low_noise_noise_1_dataset', 'Low_noise_noise_2_dataset',
                                      'Low_const_const_1_dataset', 'Low_const_const_2_dataset',
                                      'High_sin_tooth_1_dataset',
                                      'High_sin_tooth_2_dataset']
        for dataset_name in dataset_exclude_for_deploy:
            print('\n> loading data for dataset', dataset_name)
            data_img = np.load(os.path.join(machine_output_dir, 'temp', f'saliency_map_img.deploy.{dataset_name}.pt.npy'))
            data_param = np.load(os.path.join(machine_output_dir, 'temp', f'saliency_map_param.deploy.{dataset_name}.pt.npy'))
            data_pos = np.load(os.path.join(machine_output_dir, 'temp', f'saliency_map_pos.deploy.{dataset_name}.pt.npy'))

            # temp_video(data_img, 'img')
            # temp_video(data_param, 'param')
            # temp_video(data_pos, 'pos')

            saliency_heatmap_plot(data_img, data_param, data_pos, extra_title=dataset_name)
