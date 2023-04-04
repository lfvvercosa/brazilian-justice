# Steps for running the code:

## 1) Clone the git repository using https:

	git clone https://github.com/lfvvercosa/brazilian-justice.git

## 2) Download the Brazilian Justice dataset

The full dataset is available at: 

- https://data.4tu.nl/datasets/fcdc27b9-44fd-476f-9a2d-1774e96e505f/1
- https://www.kaggle.com/datasets/lfvvercosa/brazilian-justice-processes

To run this code, it is sufficient to download the following folder and files:
- 'tribunais_superiores' (folder)
- 'df_assuntos.csv' (file)
- 'df_classes.csv' (file)
- 'df_movimentos.csv' (file) 

Please, place the downloaded files inside the 'dataset' folder in the root directory. 

## 3) Install python packages contained in 'requirements.txt' file (consider using a virtual environment)

	pip3 install -r requirements.txt

## 4) Install 'graphviz' and 'xdg-utils' software on machine. They are used by PM4Py to create and show images. Example on Linux Debian:

	sudo apt-get install graphviz
	sudo apt-get install xdg-utils

## 5) Run 'main.py'.

	python3 main.py
