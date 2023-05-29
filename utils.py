# Librerías para renombrar y mover los archivos
import os
import shutil
import re
# Librerias para generar el json
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import save_json, join
import collections
# Librerias para descomprimir
import zipfile
# Librerías necesarias para la representación de la predicción
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import random

def unzip_file(source_folder, destination_folder):
    # Creamos la carpeta destino si no existe
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Abrimos el archivo zip
    with zipfile.ZipFile(source_folder, 'r') as zip_ref:
        # Extraemos todos los archivos en la carpeta destino
        zip_ref.extractall(destination_folder)

    print('Archivos descomprimidos correctamente.')


def rename_and_move_files(input_folder, output_folder_images, output_folder_labels):
    for file in os.listdir(input_folder):
        if "mask" in file:
            # Procesamos los archivos de máscara
            case_id = re.search(r'\d+', file).group()
            new_label_file = f"MS_{case_id}.nii.gz"
            shutil.copyfile(os.path.join(input_folder, file), os.path.join(output_folder_labels, new_label_file))
        else:
            # Procesamos los archivos de imagen
            case_id = re.search(r'\d+', file).group()
            time_suffix_match = re.search(r'_time(\d+)', file)

            if time_suffix_match:
                # Restamos 1 del sufijo temporal para que los números comiencen desde 0000 en lugar de 0001
                time_suffix = str(int(time_suffix_match.group(1)) - 1).zfill(4)
            else:
                print(f"No se encontró un sufijo temporal para el archivo: {file}")
                continue

            new_image_file = f"MS_{case_id}_{time_suffix}.nii.gz"
            shutil.copyfile(os.path.join(input_folder, file), os.path.join(output_folder_images, new_image_file))


def generate_training_cases_list(input_folder_images, input_folder_labels, task_folder):
    training_cases_list = []

    # Se obtiene una lista de todos los archivos de imagen en la carpeta de entrada
    image_files = os.listdir(input_folder_images)

    # Se obtiene una lista de todos los archivos de 'labels' en la carpeta de entrada
    label_files = os.listdir(input_folder_labels)

    # Creamos un diccionario para cada case_id, con case_id como clave y las rutas
    # de los archivos de imagen como valor
    case_id_to_images = collections.defaultdict(list)
    for image_file in image_files:
        case_id = re.search(r'\d+', image_file).group()
        relative_image_path = os.path.relpath(os.path.join(input_folder_images, image_file), start=task_folder)
        case_id_to_images[case_id].append(relative_image_path)

    for label_file in label_files:
        case_id = re.search(r'\d+', label_file).group()

        relative_label_path = os.path.relpath(os.path.join(input_folder_labels, label_file), start=task_folder)

        training_cases_list.append({
            "image": case_id_to_images[case_id],
            "label": [relative_label_path]
        })

    return training_cases_list


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          training_cases_list,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didnt like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
        'training': training_cases_list  # Lista de casos de entrenamiento que hemos obtenido
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


def visualize_prediction(predictions_results_path, images_patients_path, labels_patients_path):
    # Obtenemos una lista de todos los archivos en el directorio
    files = os.listdir(predictions_results_path)

    # Seleccionamos un archivo al azar de la lista
    file_selected = random.choice(files)
    print(f"El archivo seleccionado es: {file_selected}")

    # El archivo seleccionado de ser del tipo .nii.gz
    if file_selected.endswith('.nii.gz'):
        # Cargamos la imagen .nii.gz
        img_pred = nib.load(os.path.join(predictions_results_path, file_selected))
        mask_original = nib.load(os.path.join(labels_patients_path, file_selected))

        # Obtenemos el nombre del archivo original correspondiente
        original_file_name_0 = file_selected.replace('.nii.gz', '_0000.nii.gz')
        original_file_name_1 = file_selected.replace('.nii.gz', '_0001.nii.gz')

        img_original_0 = nib.load(os.path.join(images_patients_path, original_file_name_0))
        img_original_1 = nib.load(os.path.join(images_patients_path, original_file_name_1))

        # Conviertimos la imagen en una matriz numpy
        data_pred = img_pred.get_fdata()
        data_original_0 = img_original_0.get_fdata()
        data_original_1 = img_original_1.get_fdata()
        mask_original = mask_original.get_fdata()

        # Eligemos un corte a lo largo del eje z
        corte_z_pred = data_pred[:, :, data_pred.shape[2] // 2]
        corte_z_original_0 = data_original_0[:, :, data_original_0.shape[2] // 2]
        corte_z_original_1 = data_original_1[:, :, data_original_1.shape[2] // 2]
        corte_z_mask_original = mask_original[:, :, mask_original.shape[2] // 2]

        # Superponemos la predicción en la imagen original
        img_superimposed = corte_z_original_1.copy()
        img_superimposed[corte_z_pred > 0] = np.max(corte_z_original_1)

        # Mostramos la predicción
        fig = plt.figure(figsize=(12, 13))

        # Fusionamos las dos últimas subfiguras en una única
        # subfigura más grande que ocupe ambas columnas. 'subplot2grid()'
        axs = [plt.subplot2grid((3, 2), (i, j)) for i in range(2) for j in range(2)]
        axs.append(plt.subplot2grid((3, 2), (2, 0), colspan=2))

        axs[0].imshow(corte_z_original_0, cmap='gray')
        axs[0].set_title(f"Original Image\n{original_file_name_0}")

        axs[1].imshow(corte_z_original_1, cmap='gray')
        axs[1].set_title(f"Original Image\n{original_file_name_1}")

        axs[2].imshow(corte_z_mask_original, cmap='gray')
        axs[2].set_title(f"Original mask for\n{file_selected}")

        axs[3].imshow(corte_z_pred, cmap='gray')
        axs[3].set_title(f"Prediction mask for\n{file_selected}")

        # Utilizamos colormap en rojo para destacar la lesion y sea más apreciable
        axs[4].imshow(img_superimposed, cmap='gray')
        axs[4].imshow(corte_z_pred * np.max(corte_z_original_1), alpha=0.6, cmap='Reds')
        axs[4].set_title(f"Prediction nnUNet\n{file_selected}")

        plt.show()
    else:
        print("El archivo seleccionado no es .nii.gz")

