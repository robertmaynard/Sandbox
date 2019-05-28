#How to setup machine to use benchmark utlities#

#OSX and Unix#

First we need to setup a stable python platform for python, this means
getting pip the python package manager, and virtual env which isolates
projects python libraries.

```
sudo easy_install pip
sudo pip install virtualenv
```

Next we need to create a new virtual env of python for benchmark. I personally
like a folder called env in my source directory that will hold the needed python
libraries. By the way the env folder is already part of .gitignore. 

```
mkdir env
virtualenv env
````

Now we have virtual env folder setup!

Now all we have to do is setup the requirements for cfarm, which is super
easy since we are using pip. Pip provides the ability to install all the
requirements of a project ( http://www.pip-installer.org/en/latest/cookbook.html#requirements-files ).
So lets go install cfarm's requirements.


```
env/bin/pip install -r requirements.txt
```

And now your virtualenv has all of cfarm requirements and you can start
developing!

Now to setup an alias to call cfarm:
```
<path_to_cfarm>/env/bin/python <path_to_cfarm>/cfarm.py $*
```


