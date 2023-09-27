#!/bin/zsh
for i in {1..5}
do
python main.py -a ResNet -c1 Bacillus -c2 Escheri 
python main.py -a ResNet -c1 Bacillus -c2 Enterococcu
python main.py -a ResNet -c1 Bacillus -c2 Lactobacillus 
python main.py -a ResNet -c1 Bacillus -c2 Listeria 
python main.py -a ResNet -c1 Bacillus -c2 Salmonella 
python main.py -a ResNet -c1 Bacillus -c2 Staphylococcus 
python main.py -a ResNet -c1 Bacillus -c2 Pseudomonas 
python main.py -a ResNet -c2 Listeria -c1 Escheri 
python main.py -a ResNet -c1 Escheri -c2 Enterococcu 
python main.py -a ResNet -c1 Escheri -c2 Staphylococcus  
python main.py -a ResNet -c1 Escheri -c2 Lactobacillus  
python main.py -a ResNet -c1 Escheri -c2 Pseudomonas 
python main.py -a ResNet -c2 Salmonella -c1 Escheri 
python main.py -a ResNet -c1 Enterococcu -c2 Lactobacillus 
python main.py -a ResNet -c1 Enterococcu -c2 Salmonella 
python main.py -a ResNet -c1 Enterococcu -c2 Listeria 
python main.py -a ResNet -c1 Enterococcu -c2 Staphylococcus 
python main.py -a ResNet -c1 Enterococcu -c2 Pseudomonas 
python main.py -a ResNet -c1 Lactobacillus -c2 Salmonella 
python main.py -a ResNet -c1 Lactobacillus -c2 Listeria 
python main.py -a ResNet -c1 Lactobacillus -c2 Staphylococcus 
python main.py -a ResNet -c1 Lactobacillus -c2 Pseudomonas 
python main.py -a ResNet -c1 Listeria -c2 Staphylococcus 
python main.py -a ResNet -c1 Listeria -c2 Pseudomonas 
python main.py -a ResNet -c2 Salmonella -c1 Listeria 
python main.py -a ResNet -c1 Salmonella -c2 Staphylococcus 
python main.py -a ResNet -c1 Salmonella -c2 Pseudomonas 
python main.py -a ResNet -c1 Staphylococcus -c2 Pseudomonas 
done