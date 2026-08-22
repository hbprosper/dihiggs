# dihiggs
Modeling the di-Higgs cross section as a function of the parameters of the Higgs effective field theory (HEFT).


## Dependencies
The notebooks, and `dihiggs` module, in this repository depend on one or more well-known
well-engineered and Python modules.

| __modules__   | __description__     |
| :---          | :---        |
| pytorch       | a powerful, flexible, research-level machine learning toolkit |
| numpy         | array manipulation and numerical analysis   |
| matplotlib    | a widely used plotting module for producing high quality plots |
| sympy         | an excellent symbolic mathematics module |
| scipy         | scientific computing    |
| pandas        | data table manipulation, often with data loaded from csv files |
| imageio      | photo-quality image display module |
| iminuit | a rewrite of the venerable CERN minimizer Minuit |
| emcee | one of many, many, Markov chain Monte Carlo modules |
| tqdm         | progress bar |
| joblib | module to save and load Python objects |
| importlib | importing and re-importing modules |

##  Installation
The simplest way to install these Python modules is first to install a software environment system. 
You could just bite the bullet and install Anaconda! However, it may be better to install
**miniconda3**, which is a very slim version of Anaconda, on your laptop. Do so by following the instructions at:

https://www.anaconda.com/docs/getting-started/miniconda/system-requirements

Software environment systems such as Anaconda (__conda__ for short) make
it possible to have several separate self-consistent named
**environments** on a single machine, say your laptop. For example, you
may need to use Python 3.11.8 and an associated set of compatible
packages and at other times you may need to use Python 3.12.4 with
packages that require that particular version of Python.  If you install software without using environments there is
the danger that the software on your laptop will eventually become
inconsistent. Anaconda (and its lightweight companion miniconda)
provide a way, for example, to create a software environment
consistent with Python 3.11.8 and another that is consistent with
Python 3.12.4 without conflict.  

Of course, like anything humans make, miniconda3 is not
perfect. There are times when the only solution is to delete an
environment and rebuild by reinstalling the desired packages.

### Miniconda3

After installing miniconda3, it is a good idea to update conda using the command
```bash
conda update conda
```
#### Step 1 
Assuming conda is properly installed and initialized on your laptop, you can create an environment, here called *hh* using the command
```bash
conda create --name hh
```
and activate it by doing
```bash
conda activate hh
```
You need create the environment only once, but you must activate the desired environment whenever you create a new terminal window.

#### Step 2 
First install **pytorch**. (Tip: search the web for **conda install** and the
module name to get the exact syntax just in case it has changed.)
```
	conda install pytorch –c pytorch
```
The installation of **pytorch** will install some of the modules listed below, so check the list of modules as they scroll past.

#### Step 3
Install *jupyterlab*, *matplotlib*,  etc.
```bash
	conda install jupyterlab notebook
	conda install matplotlib
	conda install pandas
	conda install sympy
	conda install imageio
	conda install tqdm
```
Again be sure to check the exact syntax. This does change from time to time!

#### Step 4
Install __git__ if it is not yet on your system, then clone the **dihiggs** package.
```bash
	conda install git
	mkdir Projects
	cd Projects
	git clone https://github.com/hbprosper/dihiggs
	cd dihiggs
	pip install -e .
```
In the above the GitHub package *dihiggs* has been cloned into a folder/directory called *Projects* and installed so that it is available from any other folder on your machine.
