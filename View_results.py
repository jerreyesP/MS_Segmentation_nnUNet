import utils

# Directorio de resultados .nii.gz y archivos originales
predictions_results_path = r'prediction_test/results_prediction_2d'
images_patients_path = r'nnUNet_raw_data_base/nnUNet_raw_data/Dataset500_MS_Lesions/imagesTr'
labels_patients_path = r'nnUNet_raw_data_base/nnUNet_raw_data/Dataset500_MS_Lesions/labelsTr'

utils.visualize_prediction(predictions_results_path, images_patients_path, labels_patients_path)