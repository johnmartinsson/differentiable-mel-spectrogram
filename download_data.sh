# clone datasets
git clone https://github.com/soerenab/AudioMNIST.git
git clone https://github.com/karolpiczak/ESC-50.git

mkdir data

# move datasets to data folder
mv ESC-50/ data/esc50
mv AudioMNIST/ data/audio_mnist

# resample AudioMNIST to 8000 Hz
echo "resample all Audio-MNIST files to 8000 Hz"
for file in $(find ./data/audio_mnist -type f -name "*.wav"); do
    sox $file -r 8000 ${file%.wav}_8k.wav
    mv ${file%.wav}_8k.wav $file
done

# initialize datasets
echo "initialize audio datasets ..."
python3 init_dataset.py $(pwd)/data
