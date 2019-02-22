# Bastet
CS234 Texas Holdem AI player

## Installation Instructions

Setup virtualenv with python3.6:

`sudo pip install virtualenv`
`virtualenv .env --python=python3.6`
`source .env/bin/activate`

To exit the virtual environment at any time:
`deactivate`

intall required packages:
`pip install -r requirements.txt`

Test to see if PyPokerEngine installed correctly:
`python helloWorld.py`

Run a heads up match with FishPlayer vs RandomPlayer
`python testAImatch.py`

Run a heads up match with a human console player vs FishPlayer
`python testConsolematch.py`
