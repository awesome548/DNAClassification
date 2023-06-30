#!/bin/zsh

python main.py -a ResNet -c1 Bacillus -c2 Enterococcu
python main.py -a ResNet -c1 Bacillus -c2 Lactobacillus 
python main.py -a ResNet -c1 Bacillus -c2 Salmonella 
python main.py -a ResNet -c1 Bacillus -c2 Listeria 
python main.py -a ResNet -c1 Bacillus -c2 Staphylococcus 
python main.py -a ResNet -c1 Bacillus -c2 Escheri 

python main.py -a ResNet -c1 Enterococcu -c2 Lactobacillus 
python main.py -a ResNet -c1 Enterococcu -c2 Salmonella 
python main.py -a ResNet -c1 Enterococcu -c2 Listeria 
python main.py -a ResNet -c1 Enterococcu -c2 Staphylococcus 
python main.py -a ResNet -c1 Enterococcu -c2 Escheri 

python main.py -a ResNet -c1 Lactobacillus -c2 Salmonella 
python main.py -a ResNet -c1 Lactobacillus -c2 Listeria 
python main.py -a ResNet -c1 Lactobacillus -c2 Staphylococcus 
python main.py -a ResNet -c1 Lactobacillus -c2 Escheri 

python main.py -a ResNet -c1 Salmonella -c2 Listeria 
python main.py -a ResNet -c1 Salmonella -c2 Staphylococcus 
python main.py -a ResNet -c1 Salmonella -c2 Escheri 

python main.py -a ResNet -c1 Listeria -c2 Staphylococcus 
python main.py -a ResNet -c1 Listeria -c2 Escheri 

python main.py -a ResNet -c1 Staphylococcus -c2 Escheri 