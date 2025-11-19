import csv
import os
import pandas as pd

def create_csv(root_folder, output_csv):
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'age'])

        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)

            if os.path.isdir(folder_path):
                age = folder_name
                # List all images in the folder
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)

                    if os.path.isfile(image_path):
                        writer.writerow([image_name, age])

    # print(f"CSV file created at: {output_csv}")

root_folder = 'C:/Users/leyih/OneDrive/Desktop/8.7.2024/The Years/Y3/Sem 1/MyCNN/face_age'  # Replace with image folder path
output_csv = 'csv_dataset/kaggle_dataset.csv'  # Path to the output CSV file
create_csv(root_folder, output_csv)

# df = pd.read_csv('csv_dataset/kaggle_dataset.csv')
# print(f'Dataframe length: {len(df)}')
# print()
# print(df.head())