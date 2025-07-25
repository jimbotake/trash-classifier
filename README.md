
# Trash Classifier

## About The Project

Trash Classifier adalah aplikasi berbasis AI untuk mengklasifikasikan jenis-jenis sampah menggunakan model CNN.  
Aplikasi ini memungkinkan pengguna mengupload gambar sampah dan mendapatkan prediksi kategori sampah tersebut.

### Built With
- Python
- TensorFlow / Keras
- Streamlit

## Getting Started

### Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Installation

1. Clone repo ini:
   ```bash
   git clone https://github.com/username/trash-classifier.git
   cd trash-classifier
   ```
2. Install dependencies seperti di atas.
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Usage

- Upload gambar sampah melalui aplikasi web Streamlit.
- Aplikasi akan menampilkan prediksi kelas dan confidence level.

## Visulisasi Aplikasi
![Tampilan Aplikasi](images/capture_aplikasi.png)

## License

Distributed under the MIT License.

## Contact

Your Name - dimasony.dewantara@gmail.com  
Project Link: https://github.com/jimbotake/trash-classifier

## Dataset Citation

This project uses the [TrashNet dataset](https://github.com/garythung/trashnet) for training and evaluation.

Please cite the dataset as:

> Thung, G., & Yang, M. (2016). **TrashNet: A Dataset for Image Classification of Trash**. Retrieved from https://github.com/garythung/trashnet

