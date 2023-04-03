# Steps for running the code

## 1) Download the Brazilian Justice dataset

The full dataset is available at: 

- https://data.4tu.nl/datasets/fcdc27b9-44fd-476f-9a2d-1774e96e505f/1
- https://www.kaggle.com/datasets/lfvvercosa/brazilian-justice-processes

However, in order to run this code, it is only necessary to download the folder:
- 'tribunais_superiores' 

And the files: 
- 'df_assuntos.csv'
- 'df_classes.csv'
- 'df_movimentos.csv'. 

Please, place the folder and files inside 'dataset' project root folder. 

## 2) Install python packages contained in 'requirements.txt' file
	```console
	pip3 install -r requirements.txt
	```
## 3) Install 'graphviz' software on machine. Example on Linux Debian:
	```console
	sudo apt-get install graphviz
	```

## 4) Run 'main.py'.
	```console
	python3 main.py
	```