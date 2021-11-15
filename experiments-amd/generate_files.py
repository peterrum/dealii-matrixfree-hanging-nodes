import json
import os

def create_file(GeometryType, NRefinements, Degree, SetupOnlyFastAlgorithm, 
        TestHighOrderMapping, Categorize, VectorizationType, FileName):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/generate_files.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["GeometryType"]           = GeometryType
    datastore["NRefinements"]           = NRefinements
    datastore["Degree"]                 = Degree
    datastore["SetupOnlyFastAlgorithm"] = SetupOnlyFastAlgorithm
    datastore["TestHighOrderMapping"]   = TestHighOrderMapping
    datastore["Categorize"]             = Categorize
    datastore["VectorizationType"]      = VectorizationType

    # write data to output file
    with open(FileName, 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    create_file("annulus", 8, 1, False, False, False, "index", "run-exp1-a-1.json")
    create_file("annulus", 8, 2, False, False, False, "index", "run-exp1-a-2.json")
    create_file("annulus", 8, 3, False, False, False, "index", "run-exp1-a-3.json")
    create_file("annulus", 7, 4, False, False, False, "index", "run-exp1-a-4.json")
    create_file("annulus", 7, 5, False, False, False, "index", "run-exp1-a-5.json")
    create_file("annulus", 6, 6, False, False, False, "index", "run-exp1-a-6.json")
    
    create_file("quadrant", 7, 1, False, False, False, "index", "run-exp1-c-1.json")
    create_file("quadrant", 7, 2, False, False, False, "index", "run-exp1-c-2.json")
    create_file("quadrant", 7, 3, False, False, False, "index", "run-exp1-c-3.json")
    create_file("quadrant", 6, 4, False, False, False, "index", "run-exp1-c-4.json")
    create_file("quadrant", 6, 5, False, False, False, "index", "run-exp1-c-5.json")
    create_file("quadrant", 6, 6, False, False, False, "index", "run-exp1-c-6.json")


if __name__== "__main__":
  main()
