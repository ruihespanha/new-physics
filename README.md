# new-physics

## Installation instructions

1) Clone repository from command line

   + remove the folder "c:\Users\ruihe\GitHub\new-physics" if it already exists
   + Open a windows terminal in c:\Users\ruihe\GitHub
   + Clone the repository using
        git clone https://github.com/ruihespanha/new-physics.git

      This should create the folder "c:\Users\ruihe\GitHub\new-physics"

2) Open VS code aND Create a Python environment:

   + open the FOLDER "c:\Users\ruihe\GitHub\new-physics" (not "open file", use "open folder")
   + click on the python symbol in the left hand side (8ths from the top?)
   + It should show "No active Environment"
   + Click on "+" in froNT or "worspace environments"
   + Choose the environment name: "venv"
   + Choose the python version "3.12.10"
   + DO NOT select requirements file "requirements.txt" 

   + once the enviroment is created, click on the symbol ">_" to the right of the environement just created.
     This should open a terminal in the enviroment
     The prompt of the terminal should show the environment name in (green?)

3) Installing the required packages

   + In a terminal with the environment created above, install pytorch from the terminal using

      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

      (from https://pytorch.org/get-started/locally/)
     
   + install the rest of the packages using
        pip install -r requirements.txt

4) unzip the pickle files into "C:\Users\ruihe\GitHub\new-physics\Training_data_pickle"

You should be good to run the script "C:\Users\ruihe\GitHub\new-physics\Basic simulations\AISims\Final\CNN_with_difference.py"
