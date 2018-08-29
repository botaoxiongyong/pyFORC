# pyFORC
this is for first order reversal curves
inlcuding irregualr forc, and remforc (developing)

# Installation
1. Clone this repository with Git:

        $ git clone https://github.com/botaoxiongyong/pyFORC
2. install dependences:

        $ cd pyFORC
        $ pip3 install -r requirements.txt
3. install pyFORC

        # in the same derectory
        $ python3 setup.py install
        
# Runing

        $ python3 -m pyFORC_log /data_directory/irgular_forc_file.txt 5
        #5 is the default smooth factor, change by yourself.
