# new-physics

## Installation instructions

1) Clone repository from command line

   + remove the folder "c:\Users\ruihe\GitHub\new-physics" if it already exists
   + Open terminal in c:\Users\ruihe\GitHub
   + Type
        git clone https://github.com/ruihespanha/new-physics.git

      This should create the folder "c:\Users\ruihe\GitHub\new-physics"

2) Create Python environment from VS code

   + open the FOLDER "c:\Users\ruihe\GitHub\new-physics" (not "open file", use "open folder")
   + In VS code, click on the python symbol in the left hand side (8ths from the top?)
   + It should show "No active Environment"
   + Click on "+" in from or "worspace environments"
   + Choose the environment name: "venv"
   + Choose the python version "3.12.10"
   + DO NOT select requirements file "requirements.txt" 

   + once the enviroment is created, click on the , click on the symbol ">_" which should open a terminal in the enviroment
    The prompt of the terminal should show the environment name in (green?)

3) Installing the required packages

   + In a terminal with the environment created above, install pytorch from the terminal using

      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

   + install the rest of the packages using
        pip install -r requirements.txt

4) unzip the pickle files into "C:\Users\ruihe\GitHub\new-physics\Training_data_pickle"

You should be good to run the scripts