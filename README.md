# Investigating Bias and Fairness in Appearance-based Gaze Estimation

This repository contains the annotations and code, and provides links to the trained models, accompanying our paper *"Investigating Bias and Fairness in Appearance-based Gaze Estimation"*, the first extensive evaluation of fairness in appearance-based gaze estimation across ethnicity and gender attributes.

## Acknowledgements / Third-Party Code

This repository includes modified versions of the following open-source projects, used for training and evaluating the gaze estimation models:

- [CrossGaze](https://github.com/AndyCatruna/CrossGaze)
- [GazeTR](https://github.com/yihuacheng/GazeTR)
- [L2CS-Net](https://github.com/ahmednull/l2cs-net)
- [MCGaze](https://github.com/zgchen33/MCGaze)
- [PureGaze](https://github.com/yihuacheng/PureGaze)

## Dataset Labeling

The gaze estimation datasets used in this work (**Gaze360** and **GazeCapture**) do not come with demographic labels. We therefore annotated both datasets with ethnicity and gender sensitive attributes using the following procedure:

1. **Automatic annotation.** We used [FairFace](https://github.com/dchen236/FairFace), a deep learning model for facial attribute estimation, to obtain an initial ethnicity and gender prediction for each subject's face crop.
2. **Manual verification.** Every automatic annotation was then reviewed by a human annotator.
   - If the automatic annotation and the human annotator's judgment agreed, the sample was labeled with that class.
   - If the annotator was unsure of the correct label, the sample was labeled as **"other"**.

**Labels used:**
- **Ethnicity (4 classes):** `caucasian`, `asian`, `afro_american`, `other`
- **Gender (3 classes):** `male`, `female`, `other`

The final labels for both datasets are provided in this repository:
- `Gaze360_Labels/`
- `GazeCapture_Labels/`

## Dataset Preprocessing

For each gaze estimation model evaluated in this work (CrossGaze, MCGaze, L2CS-Net, PureGaze, GazeTR), we used the preprocessing pipeline defined by the model's original implementation (e.g., face/eye detection, cropping, and normalization steps specific to each method).

- **Gaze360** was used for both **training** and **testing** the models.
- **GazeCapture** was used exclusively for **testing**.

### Gaze360 Preprocessing

For **Gaze360**, we used the dataset's respective preprocessing code to prepare the data for training and evaluation.

### GazeCapture Preprocessing

For **GazeCapture**, we used the [faze_preprocess](https://github.com/swook/faze_preprocess/tree/master) repository to convert the original 2D gaze annotations into 3D gaze annotations. We further modified the preprocessing code to convert GazeCapture into a Gaze360-like data structure, allowing it to be used consistently with our evaluation pipeline. The modified preprocessing scripts are provided in the ```GazeCapturePreprocess/``` folder.

Download the following [file](https://huggingface.co/akgulburak/gaze-estimation-diversity/blob/main/GazeCapture_supplementary.h5), GazeCapture_supplementary.h5 [from https://huggingface.co/akgulburak/gaze-estimation-diversity] and place it in ```GazeCapturePreprocess/``` to run the preprocessing code.

## Model Download

The trained models used in our experiments can be downloaded from the following link:

[`Huggingface Models Link`](https://huggingface.co/akgulburak/gaze-estimation-diversity)

## Results from the Paper

A brief summary of our key findings:

- **Dataset bias.** Both Gaze360 and GazeCapture are heavily imbalanced across ethnicity groups (Caucasian samples dominate), while gender is relatively balanced in both datasets.
- **Model bias.** All five evaluated models (CrossGaze, MCGaze, L2CS-Net, PureGaze, GazeTR) show statistically significant fairness disparities across ethnicity, with the strongest bias observed for the Caucasian–Afro-American pair. Gender-related disparities are generally smaller but still present in most models.
- **Bias mitigation.** We benchmarked two pre-processing methods (oversampling, resampling) and one in-processing method (loss reweighting). None of the methods consistently improved fairness across all models, datasets, and attributes - some even increased bias. Resampling and loss reweighting were the most effective on average, while PureGaze benefited most consistently from mitigation.
- **Takeaway.** Standard bias mitigation techniques developed for classification/regression tasks provide only limited fairness gains in gaze estimation, motivating the need for bias mitigation strategies tailored for the gaze estimation task.

For full quantitative results (Wasserstein distance, KS distance, t/p-values across all models, datasets, and demographic pairs), please refer to the paper.

## Fairness Calculation

To calculate the fairness metrics reported in the paper, run:

```bash
uv run fairness.py --input <input_folder_with_csv_files> --output <output_excel_file_name> --mode [ethnicity | gender]
```

## License

This project is licensed under CC BY-NC-SA 4.0 - see [LICENSE](LICENSE) for details.

## Citation

Please cite our work as:

```
@inproceedings{akgul2026biasingaze,
  title     = {Investigating Bias and Fairness in Appearance-based Gaze Estimation},
  author    = {Akg{\"u}l, Burak and {\c{S}}ahin, Erol and Kalkan, Sinan},
  booktitle = {2026 IEEE 20th International Conference on Automatic Face and Gesture Recognition (FG)},
  year      = {2026},
  address   = {Kyoto, Japan},
  publisher = {IEEE},
  doi       = {10.1109/FG67764.2026.11557007}
}
```