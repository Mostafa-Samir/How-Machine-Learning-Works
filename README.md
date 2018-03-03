# Unnamed Upcoming Book

This repository contains the code accompanying the work done in [Unnamed Upcoming Book](link-tbd) by _Mostafa Samir_, Manning Publications. All the code is written with `python`.

## Usage
To start using this code, you need first to clone it to your local machine. There are two ways you can do that:

1. Using the terminal/cmd (provided that you have `git` installed) by running:
```
$ git clone https://github.com/Mostafa-Samir/Unnamed-Upcoming-Book-Code.git
```

2. By clicking on the green button above labeled **Clone or download** and choosing **Download ZIP**. Extract the zip after downloading it.

If you followed the way of the book (described in Appendix A) and used the Anaconda distribution; all you need to do is to to enter the repository's directory on your machine and run (through the terminal/cmd):
```
$ jupyter notebook
```

The Jupyter Notebook interface should open on your default browser and everything will be installed for the code to work.

However, if you chose to run your own environment using `pip` or `conda` in a virtual environment, then you're on your own installing the required packages! We're not worried though, you probably know what you're doing :wink:

## Structure
The repository is structured into directories for each part of the book. Within each part, every chapter has its own notebook. In addition to chapters' notebooks, you may find a `utility.py` file in the part's directory; this file contains utility methods used in the notebook but its implementation may be outside the scope of the book at that phase.

Moreover, all the datasets we use in the book can be found in the `datasets` directory.

## Errors and Fixes
Your error reports and/or fixes are very much welcomed! If you found an error, please do open an issue here describing the error, the environment you had the error on, and how the error can be reproduced (if applicable).

If you know how the error can be fixed, it will much appreciated if you forked the repo, fixed the error in your fork and opened a pull request to merge the fix back in here :+1:
