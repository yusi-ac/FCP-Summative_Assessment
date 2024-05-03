FCP-Summative_Assessment README.txt

Group_28
Github repo Link: https://github.com/yusi-ac/FCP-Summative_Assessment

Requirements:

Install python latest version 
Install the required libraries: numpy, matplotlib, sys, argparse
pip install numpy
pip install matplotlib
pip install sys
pip install argparse

Type in the command to run the code for each task:
Task 1: 
	$ python3 assignment.py -ising_model 
	$ python3 assignment.py -ising_model -external -0.1 
	$ python3 assignment.py -ising_model -alpha 10 
	$ python3 assignment.py -test_ising 

Task 2:
	$ python3 assignment.py -defuant 
	$ python3 assignment.py -defuant -beta 0.1 
	$ python3 assignment.py -defuant -threshold 0.3 
	$ python3 assignment.py -test_defuant 
OR：
Testing:
You can run the following command to test your code:
python assignment.py -test_defuant -threshold 0.2 -beta 0.2 -num_people 50

Run the defuant model
python assignment.py -defuant -threshold 0.2 -beta 0.2 -num_people 50 
# A defuant model on a 1D grid will be generated, using the relevant parameters you have given.


Task 3:
	$ python3 assignment.py -network 10 
	Mean degree: <number>
	Average path length: <number>
	Clustering co-efficient: <number>
	$ python3 assignment.py -test_network 

Task 4:
	$ python3 assignment.py -ring_network 10 
	$ python3 assignment.py -small_world 10 
	$ python3 assignment.py -small_world 10 -re_wire 0.1 

Task 5:
	$ python3 assignment.py -defuant -use_network 10 
