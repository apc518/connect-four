# Connect 4 Bot

A minimax connect four player with a-b pruning, move ordering, and
optional multithreading and thread pruning.

### Project Structure
`connect4.py` -- authored by Adam A. Smith, drives the game either through a GUI or CLI. Not my code.

`connect4player.py` -- authored by me, Andy Chamberlain, provides the class `ComputerPlayer` and the method `ComputerPlayer.pick_move`

`README.md` -- The file you are currently reading

`requirements.txt` -- Stores the pip requirements for the project. See the setup section below for usage.

### Setup

```
git clone https://github.com/apc518/connect-four.git
cd connect-four
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python connect4.py
```

For help with command line args to connect4.py, use `python connect4.py -h`
