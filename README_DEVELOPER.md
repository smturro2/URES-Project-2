## Setup Python Environment
This python application is built,tested, and hosted around python 3.9. 
For best results [install this version of python](https://www.python.org/downloads/). 

1. **Clone** the repo to your local computer


2. **Create a virtual environment** by typing the following in the command prompt. Make sure you are at the top most level of the repo.

         py -m venv venv
3. **Activate virtual environment**

        venv\Scripts\activate.bat

4. Install the **requirements**.

         pip install -r requirements.txt

### Troubleshooting Python Setup

- *'py' is not recognized*. 
  - Depending on how you downloaded python your PATH variable for python (the _py_ in this walkthrough) may be diffrent.
Other common names are _python_ or _python3_. Its whatever you type into the command prompt in order to start python. 
  Replace the _py_ with whatever your PATH variable for python is. Follow [this tutorial](https://www.educative.io/edpresso/how-to-add-python-to-path-variable-in-window) 
  if you can't figure out your python's PATH variable and you know for certain python is installed.

- *'pip' is not recognized*. 
  - pip is not defined as a PATH variable. Instead of just using *pip*
use *py -m pip*. It is rare but if you downloaded python in a weird way you may not have pip installed. You'll have to search for how to install pip on your computer. 
The process differs depending on if you're using PC/mac/ubuntu.
    
